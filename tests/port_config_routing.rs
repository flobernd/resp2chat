//! HTTP routing behaviors for the config port (claude-relay `test_config.py`),
//! the gateway-driving half. The pure config/TOML resolution tests live in the
//! sibling `tests/port_config.rs`; this file drives the full gateway with
//! wiremock upstreams and asserts served-model resolution and per-model leaf
//! finalization.
//!
//! Glob/profile route DISPATCH tests land in Task 5 once profile resolution owns
//! routing; this file currently covers the model-fallback response headers and
//! the per-model reasoning-effort map at the upstream leaf.

mod common;

use common::config_from_yaml;
use serde_json::json;
use tower::ServiceExt;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::body_partial_json;
use wiremock::matchers::method;
use wiremock::matchers::path;

use axum::body::Body;
use http::Request;

/// Minimal single-chunk chat-completions SSE body for a wiremock upstream.
fn chat_sse_body(id: &str, content: &str) -> String {
    let chunk = json!({
        "id": id,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": null
        }],
        "usage": null
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

/// Mount a `/v1/models` catalog on `server` exposing exactly `ids`.
async fn mount_models_catalog(server: &MockServer, ids: &[&str]) {
    let data: Vec<_> = ids.iter().map(|id| json!({ "id": id })).collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": data })))
        .mount(server)
        .await;
}

/// Mount an UNCONDITIONAL streaming `/v1/chat/completions` target that always
/// answers with `chat_sse_body("chat", label)`.
async fn mount_chat_target(server: &MockServer, label: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(chat_sse_body("chat", label)),
        )
        .mount(server)
        .await;
}

// ---------------------------------------------------------------------------
// Per-request model-resolution headers (debug aid for model mismatches)
// ---------------------------------------------------------------------------

/// A request for a model the backend does not serve falls back to the loaded
/// model, and the response is tagged with `x-llmconduit-model` (served) plus
/// `x-llmconduit-requested` (the original) so the mismatch is visible per
/// request — un-throttled, unlike the engine WARN. An exact match tags only the
/// served model and omits `x-llmconduit-requested`.
#[tokio::test]
async fn response_headers_expose_model_fallback() {
    let backend = MockServer::start().await;
    mount_models_catalog(&backend, &["served-model"]).await;
    mount_chat_target(&backend, "ok").await;

    let config = config_from_yaml(&format!(
        "upstreams:\n  - name: \"local\"\n    upstream_base_url: \"{}/v1/\"\n",
        backend.uri()
    ));

    let header = |response: &axum::response::Response, name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let request = |model: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model,
                    "stream": false,
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .expect("request")
    };

    // Mismatch: requested model is not served -> fallback + BOTH headers.
    let mismatch = llmconduit::build_app(config.clone())
        .oneshot(request("claude-opus-4"))
        .await
        .expect("response");
    assert_eq!(mismatch.status().as_u16(), 200);
    assert_eq!(
        header(&mismatch, "x-llmconduit-model").as_deref(),
        Some("served-model")
    );
    assert_eq!(
        header(&mismatch, "x-llmconduit-requested").as_deref(),
        Some("claude-opus-4")
    );

    // Exact match: served model requested -> served tag only, no `requested`.
    let exact = llmconduit::build_app(config)
        .oneshot(request("served-model"))
        .await
        .expect("response");
    assert_eq!(exact.status().as_u16(), 200);
    assert_eq!(
        header(&exact, "x-llmconduit-model").as_deref(),
        Some("served-model")
    );
    assert_eq!(header(&exact, "x-llmconduit-requested"), None);
}

// ---------------------------------------------------------------------------
// Per-model reasoning-effort map (applied at the upstream leaf)
// ---------------------------------------------------------------------------

/// End-to-end through the REAL upstream leaf: a request whose effort maps via a
/// model profile's `reasoning_effort_map` reaches the backend as
/// `chat_template_kwargs.reasoning_effort`, against the FINAL resolved model.
/// The POST mock only fires when the body carries the mapped knob, so a 200 (vs
/// wiremock's 404 on no-match) proves the map was applied at the leaf.
#[tokio::test]
async fn reasoning_effort_map_reaches_backend_chat_template_kwargs() {
    let backend = MockServer::start().await;
    mount_models_catalog(&backend, &["GLM-test"]).await;
    // Only matches when the leaf placed the mapped effort in chat_template_kwargs
    // AND resolved the request model to the served backend model.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "GLM-test",
            "chat_template_kwargs": {"reasoning_effort": "high"}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(chat_sse_body("chat-glm", "ok")),
        )
        .mount(&backend)
        .await;

    let config = config_from_yaml(&format!(
        r#"
upstreams:
  - name: "backend"
    upstream_base_url: "{}/v1/"
model_profiles:
  GLM-test:
    reasoning_effort_default: max
    reasoning_effort_map:
      high: {{ chat_template_kwargs: {{ reasoning_effort: high }} }}
      max: {{ chat_template_kwargs: {{ reasoning_effort: max }} }}
"#,
        backend.uri()
    ));

    // Anthropic request with Claude Code's adaptive-thinking + output_config.effort=high,
    // model unserved by the backend so it falls back to GLM-test.
    let app = llmconduit::build_app(config);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(
                    json!({
                        "model": "claude-opus-4",
                        "max_tokens": 16,
                        "stream": false,
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": "high"},
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(
        response.status().as_u16(),
        200,
        "leaf must POST chat_template_kwargs.reasoning_effort=high for the resolved GLM-test model"
    );
}
