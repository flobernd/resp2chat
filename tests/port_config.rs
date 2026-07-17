//! Ported surface: config-loading (gap G7, claude-relay `test_config.py`).
//!
//! claude-relay routed requests through a `model_routes` map (name -> backend
//! URL + upstream model), supported glob route keys (`claude-opus-*`), a
//! `--model-route NAME=URL[,UPSTREAM_MODEL]` CLI flag, and a TOML config file.
//! llmconduit's native routing is catalog-driven (`canonical_model_key` over a
//! YAML `upstreams` catalog), so these behaviors are adapted rather than
//! transliterated:
//!
//! - A `model_routes` entry becomes a synthetic routing upstream matched by
//!   request-model NAME (possibly a glob), slotted in `RoutingModelCatalog::resolve`
//!   strictly between an exact catalog model id and the canonical-key/default
//!   fallbacks. PRECEDENCE: exact id > exact route > glob route > canonical key >
//!   default — so an exact model id always beats an overlapping glob, matching
//!   AGENTS.md "Exact model id wins" and claude-relay's exact-before-glob-before-
//!   default.
//! - `--model-route` parses to a persisted route merged AFTER the file and env
//!   (CLI wins); a malformed spec is a clean startup `Err`, never a panic.
//! - `.toml` config loads with identical semantics to the equivalent YAML
//!   (`PersistedConfig` round-trips through both deserializers).
//! - `template_family` (G2) still resolves through the profile chain — a
//!   regression guard, since G7 touches the same config-resolution code.
//!
//! These are the PURE config-resolution tests (route parsing, precedence, TOML
//! loading, serde round-trips). The gateway-driving HTTP routing tests — which
//! mount wiremock upstreams and assert which one received the routed request —
//! live in the sibling `tests/port_config_routing.rs`.

mod common;

use common::config_from_yaml;
use common::test_config;
use llmconduit::config::Config;
use llmconduit::config::OrderedModelProfiles;
use llmconduit::config::OrderedModelRoutes;
use llmconduit::config::PersistedConfig;
use llmconduit::config::PersistedModelProfile;
use llmconduit::config::PersistedModelRoute;
use llmconduit::config::compile_model_glob;
use llmconduit::config::load_persisted_config;
use llmconduit::config::parse_model_route_spec;
use llmconduit::config::write_persisted_config;

// ---------------------------------------------------------------------------
// Unit tests: route parsing + precedence + CLI + TOML + template_family guard
// ---------------------------------------------------------------------------

/// claude-relay `test_parse_model_route_cli_spec`: `NAME=URL,UPSTREAM_MODEL`
/// parses into a route with the URL and upstream model.
#[test]
fn parses_model_route_cli_spec() {
    let (name, route) = parse_model_route_spec("claude-haiku=http://haiku:8000,Qwen2.5")
        .expect("valid model route spec");
    assert_eq!(name, "claude-haiku");
    assert_eq!(
        route.upstream_base_url.as_deref(),
        Some("http://haiku:8000")
    );
    assert_eq!(route.upstream_model.as_deref(), Some("Qwen2.5"));
}

/// The upstream model is optional: `NAME=URL` parses with no model rewrite.
#[test]
fn parses_model_route_cli_spec_without_upstream_model() {
    let (name, route) =
        parse_model_route_spec("claude-opus-*=http://opus:8000").expect("valid route spec");
    assert_eq!(name, "claude-opus-*");
    assert_eq!(route.upstream_base_url.as_deref(), Some("http://opus:8000"));
    assert_eq!(route.upstream_model, None);
}

/// A malformed `--model-route` spec is a clean `Err` (no `=`, blank name, blank
/// URL, or an unparseable URL) rather than a panic.
#[test]
fn malformed_model_route_specs_return_clean_errors() {
    // Missing `=`.
    assert!(parse_model_route_spec("claude-haiku").is_err());
    // Blank name.
    assert!(parse_model_route_spec("=http://haiku:8000").is_err());
    // Blank URL.
    assert!(parse_model_route_spec("claude-haiku=").is_err());
    // Unparseable URL.
    assert!(parse_model_route_spec("claude-haiku=not a url").is_err());
}

/// A malformed CLI route surfaces from `Config::from_env_file_and_routes` as a
/// startup `Err`, never a panic — the wiring path used by `main`.
#[test]
fn malformed_cli_route_fails_config_resolution_cleanly() {
    let result = Config::from_env_file_and_routes(None, &["bogus-no-equals".to_string()]);
    assert!(
        result.is_err(),
        "malformed --model-route must be a clean Err"
    );
}

/// A `model_routes` table entry with a missing `upstream_base_url` is a clean
/// config error, not a panic.
#[test]
fn route_missing_base_url_is_config_error() {
    let persisted = PersistedConfig {
        model_routes: OrderedModelRoutes(vec![(
            "claude-x".to_string(),
            // Table with no URL => missing upstream_base_url.
            serde_yaml::from_str("upstream_model: Kimi-K2.6").expect("route table"),
        )]),
        ..PersistedConfig::default()
    };
    assert!(Config::from_persisted(&persisted).is_err());
}

/// Glob routes compile and are flagged; literal names are not. (Internal shape
/// check that mirrors claude-relay `_is_glob_pattern` classification.)
#[test]
fn resolves_glob_and_literal_routes() {
    let config = config_from_yaml(
        r#"
model_routes:
  "claude-opus-*":
    upstream_base_url: "http://opus:8000/v1"
    upstream_model: "Kimi-K2.6"
  "claude-3-5-sonnet":
    upstream_base_url: "http://sonnet:8000/v1"
    upstream_model: "Qwen3.5"
"#,
    );
    assert_eq!(config.model_routes.len(), 2);
    let glob = config
        .model_routes
        .iter()
        .find(|route| route.name == "claude-opus-*")
        .expect("glob route");
    assert!(glob.glob.is_some(), "glob route must compile a matcher");
    assert_eq!(glob.upstream_model.as_deref(), Some("Kimi-K2.6"));
    let literal = config
        .model_routes
        .iter()
        .find(|route| route.name == "claude-3-5-sonnet")
        .expect("literal route");
    assert!(literal.glob.is_none(), "literal route has no glob matcher");
}

/// A bare-string route value is coerced to a URL-only route, mirroring
/// claude-relay's str-or-table `_coerce_model_route`.
#[test]
fn route_bare_string_value_coerces_to_url() {
    let config = config_from_yaml(
        r#"
model_routes:
  "claude-haiku": "http://haiku:8000/v1"
"#,
    );
    assert_eq!(config.model_routes.len(), 1);
    let route = &config.model_routes[0];
    assert_eq!(route.name, "claude-haiku");
    assert_eq!(route.upstream_base_url.as_str(), "http://haiku:8000/v1");
    assert_eq!(route.upstream_model, None);
}

/// CLI `--model-route` merges AFTER the file and wins on a NAME conflict
/// (documented resolution order: env/file < CLI).
#[test]
fn cli_route_overrides_file_route_with_same_name() {
    let yaml = r#"
model_routes:
  "claude-haiku":
    upstream_base_url: "http://from-file:8000/v1"
    upstream_model: "FileModel"
"#;
    let path = std::env::temp_dir().join(format!(
        "llmconduit-g7-{}.yaml",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&path, yaml).expect("write yaml");
    let config = Config::from_env_file_and_routes(
        Some(&path),
        &["claude-haiku=http://from-cli:8000/v1,CliModel".to_string()],
    )
    .expect("config");
    let _ = std::fs::remove_file(&path);

    let route = config
        .model_routes
        .iter()
        .find(|route| route.name == "claude-haiku")
        .expect("merged route");
    assert_eq!(route.upstream_base_url.as_str(), "http://from-cli:8000/v1");
    assert_eq!(route.upstream_model.as_deref(), Some("CliModel"));
}

/// A CLI `--model-route` overrides a file route whose name differs ONLY in case,
/// IN PLACE (position preserved) — `upsert` must compare names case-insensitively
/// (trim + ASCII-case-insensitive), identical to route dispatch. Without this a
/// `Claude-*` file route would shadow a `claude-*` CLI route, contradicting
/// CLI-wins semantics.
#[test]
fn cli_route_overrides_file_route_case_insensitively_in_place() {
    let yaml = r#"
model_routes:
  "alpha-model":
    upstream_base_url: "http://alpha:8000/v1"
  "Claude-*":
    upstream_base_url: "http://from-file:8000/v1"
    upstream_model: "FileModel"
"#;
    let path = std::env::temp_dir().join(format!(
        "llmconduit-g7-ci-{}.yaml",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&path, yaml).expect("write yaml");
    // CLI route name differs only in case from the file route.
    let config = Config::from_env_file_and_routes(
        Some(&path),
        &["claude-*=http://from-cli:8000/v1,CliModel".to_string()],
    )
    .expect("config");
    let _ = std::fs::remove_file(&path);

    // The case-variant CLI route must REPLACE the file route, not add a second
    // entry: still exactly two routes total.
    assert_eq!(
        config.model_routes.len(),
        2,
        "a case-only-different CLI route must override in place, not duplicate"
    );
    // Position preserved: `alpha-model` first, the overridden glob second.
    let names: Vec<&str> = config
        .model_routes
        .iter()
        .map(|route| route.name.as_str())
        .collect();
    assert_eq!(names, vec!["alpha-model", "claude-*"]);
    // The overridden route resolves to the CLI value.
    let glob = &config.model_routes[1];
    assert!(glob.glob.is_some(), "still a glob route");
    assert_eq!(glob.upstream_base_url.as_str(), "http://from-cli:8000/v1");
    assert_eq!(glob.upstream_model.as_deref(), Some("CliModel"));
}

/// A `.toml` config deserializes into the SAME `PersistedConfig` as the
/// equivalent YAML (claude-relay `test_load_proxy_config_reads_toml_routes`,
/// adapted to llmconduit's schema).
#[test]
fn toml_config_loads_identically_to_yaml() {
    let yaml = r#"
bind_addr: "127.0.0.1:4010"
upstream_base_url: "http://default:8000/v1"
request_timeout_secs: 120
flatten_content: false
model_routes:
  "claude-3-5-sonnet":
    upstream_base_url: "http://sonnet:8000/v1"
    upstream_model: "Qwen3.5"
  "claude-opus-*":
    upstream_base_url: "http://opus:8000/v1"
    upstream_model: "Kimi-K2.6"
"#;
    let toml = r#"
bind_addr = "127.0.0.1:4010"
upstream_base_url = "http://default:8000/v1"
request_timeout_secs = 120
flatten_content = false

[model_routes."claude-3-5-sonnet"]
upstream_base_url = "http://sonnet:8000/v1"
upstream_model = "Qwen3.5"

[model_routes."claude-opus-*"]
upstream_base_url = "http://opus:8000/v1"
upstream_model = "Kimi-K2.6"
"#;

    let from_yaml: PersistedConfig = serde_yaml::from_str(yaml).expect("yaml");
    let from_toml: PersistedConfig = toml::from_str(toml).expect("toml");
    assert_eq!(
        from_yaml, from_toml,
        "TOML and YAML must deserialize to an identical PersistedConfig"
    );
}

/// The `.toml` extension is detected by `load_persisted_config` (byte-identical
/// to loading the equivalent YAML through the file path).
#[test]
fn toml_file_extension_is_detected_on_load() {
    let toml = r#"
upstream_base_url = "http://default:8000/v1"

[model_routes."claude-opus-*"]
upstream_base_url = "http://opus:8000/v1"
upstream_model = "Kimi-K2.6"
"#;
    let path = std::env::temp_dir().join(format!(
        "llmconduit-g7-{}.toml",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&path, toml).expect("write toml");
    let config = Config::from_env_and_file(Some(&path)).expect("load toml config");
    let _ = std::fs::remove_file(&path);

    assert_eq!(config.upstream_base_url.as_str(), "http://default:8000/v1");
    assert_eq!(config.model_routes.len(), 1);
    assert_eq!(config.model_routes[0].name, "claude-opus-*");
    assert_eq!(
        config.model_routes[0].upstream_model.as_deref(),
        Some("Kimi-K2.6")
    );
}

/// Regression guard for G2: `template_family` still resolves through the profile
/// chain after the G7 config changes. Exercised through the PUBLIC upstream-leaf
/// seam (`finalize_request_for_backend`) — the path production runs — since the
/// per-request resolution lives at the leaf, not on `Config`. The per-model
/// `template_family` policy wins over the global override; an unmatched model
/// falls back to the global value. The profiled model carries a NON-family name
/// so the per-model `kimi` override (not name sniffing) is what drives injection.
#[test]
fn template_family_still_resolves_through_profile_chain() {
    let config = config_from_yaml(
        r#"
template_family: deepseek
model_profiles:
  "Router-X":
    template_family: kimi
"#,
    );
    // Per-model `kimi` override -> Kimi `chat_template_kwargs` injected on the
    // wire for `Router-X`, despite its non-Kimi name.
    assert_eq!(
        leaf_family_chat_template_kwargs(&config, "Router-X"),
        serde_json::json!({"thinking": true, "preserve_thinking": true})
    );
    // No per-model policy -> global `deepseek` override applies (`enable_thinking`).
    assert_eq!(
        leaf_family_chat_template_kwargs(&config, "plain-model"),
        serde_json::json!({"enable_thinking": true})
    );
}

/// Resolve the family `chat_template_kwargs` the upstream LEAF injects for
/// `backend_model`, via the PUBLIC seam: build the SAME finalization policies
/// production builds (`BackendFinalizationPolicies::from_config`) and apply them
/// through `finalize_request_for_backend` to an empty wire request. Returns the
/// injected `chat_template_kwargs` object.
fn leaf_family_chat_template_kwargs(config: &Config, backend_model: &str) -> serde_json::Value {
    use llmconduit::models::chat::ChatCompletionRequest;
    use llmconduit::upstream::BackendChatRequest;
    use llmconduit::upstream::BackendFinalizationPolicies;
    use llmconduit::upstream::finalize_request_for_backend;

    let request = ChatCompletionRequest {
        model: backend_model.to_string(),
        messages: Vec::new(),
        stream: true,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: Some(false),
        reasoning_effort: None,
        response_format: None,
        stream_options: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        extra_body: std::collections::BTreeMap::new(),
    };
    let policies = BackendFinalizationPolicies::from_config(config);
    let mut backend = BackendChatRequest::new(request, None, None, None);
    finalize_request_for_backend(&mut backend, &policies);
    backend
        .request
        .extra_body
        .get("chat_template_kwargs")
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

/// `test_config` (no routes) still behaves as a non-routing gateway: this proves
/// the routing-mode trigger does not engage without `upstreams`/`model_routes`.
#[test]
fn empty_model_routes_does_not_engage_routing_mode() {
    let config = test_config();
    assert!(config.model_routes.is_empty());
    assert!(config.upstreams.is_empty());
}

// ---------------------------------------------------------------------------
// Declaration-order regression (overlapping globs): order, not sorting, decides
// ---------------------------------------------------------------------------

/// Resolved `model_routes` preserve DECLARATION order, not alphabetical order.
/// `claude-opus-*` sorts AFTER `claude-*` alphabetically, so a `BTreeMap` would
/// reorder them; the ordered structure keeps file order. Reversing the YAML
/// flips the resolved order, proving declaration order (not sorting) decides.
#[test]
fn overlapping_glob_routes_preserve_declaration_order_not_alphabetical() {
    // `claude-opus-*` declared FIRST, even though it sorts after `claude-*`.
    let opus_first = config_from_yaml(
        r#"
model_routes:
  "claude-opus-*":
    upstream_base_url: "http://opus:8000/v1"
  "claude-*":
    upstream_base_url: "http://star:8000/v1"
"#,
    );
    let names: Vec<&str> = opus_first
        .model_routes
        .iter()
        .map(|route| route.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["claude-opus-*", "claude-*"],
        "declaration order must be preserved (a BTreeMap would sort claude-* first)"
    );

    // Reversed declaration => reversed resolved order.
    let star_first = config_from_yaml(
        r#"
model_routes:
  "claude-*":
    upstream_base_url: "http://star:8000/v1"
  "claude-opus-*":
    upstream_base_url: "http://opus:8000/v1"
"#,
    );
    let names: Vec<&str> = star_first
        .model_routes
        .iter()
        .map(|route| route.name.as_str())
        .collect();
    assert_eq!(names, vec!["claude-*", "claude-opus-*"]);
}

// ---------------------------------------------------------------------------
// serde round-trip + duplicate-key handling for OrderedModelRoutes
// ---------------------------------------------------------------------------

/// `model_routes` written by `write_persisted_config` reload through
/// `load_persisted_config` as a MAP, preserving values AND declaration order.
/// Guards against the round-trip break where the routes serialized as a YAML
/// sequence but only a map deserializes.
#[test]
fn model_routes_round_trip_through_write_and_reload_as_map() {
    let config = PersistedConfig {
        // Declared in non-alphabetical order to also lock order across the trip.
        model_routes: OrderedModelRoutes(vec![
            (
                "claude-opus-*".to_string(),
                PersistedModelRoute {
                    upstream_base_url: Some("http://opus:8000/v1".to_string()),
                    upstream_model: Some("Kimi-K2.6".to_string()),
                },
            ),
            (
                "claude-haiku".to_string(),
                PersistedModelRoute {
                    upstream_base_url: Some("http://haiku:8000/v1".to_string()),
                    upstream_model: None,
                },
            ),
        ]),
        ..PersistedConfig::default()
    };

    // The serialized form must be a MAP (`name: route`), not a sequence.
    let yaml = serde_yaml::to_string(&config).expect("serialize config");
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("parse yaml");
    assert!(
        parsed["model_routes"].is_mapping(),
        "model_routes must serialize as a YAML map, not a sequence:\n{yaml}"
    );

    let path = std::env::temp_dir().join(format!(
        "llmconduit-g7-rt-{}.yaml",
        uuid::Uuid::new_v4().simple()
    ));
    write_persisted_config(&path, &config).expect("write config");
    let reloaded = load_persisted_config(&path).expect("reload config");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        reloaded, config,
        "routes must round-trip through write + reload unchanged"
    );
    let names: Vec<&str> = reloaded
        .model_routes
        .0
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["claude-opus-*", "claude-haiku"],
        "declaration order must survive the round trip"
    );
}

/// Duplicate route keys collapse to last-wins (later value replaces the first,
/// preserving the first position), matching CLI-override and claude-relay dict
/// semantics — not silently keeping a shadowed first entry.
#[test]
fn duplicate_route_keys_collapse_to_last_wins() {
    let config = config_from_yaml(
        r#"
model_routes:
  "claude-x":
    upstream_base_url: "http://first:8000/v1"
    upstream_model: "FirstModel"
  "claude-x":
    upstream_base_url: "http://second:8000/v1"
    upstream_model: "SecondModel"
"#,
    );

    assert_eq!(
        config.model_routes.len(),
        1,
        "duplicate keys must collapse to a single route"
    );
    let route = &config.model_routes[0];
    assert_eq!(route.name, "claude-x");
    assert_eq!(
        route.upstream_base_url.as_str(),
        "http://second:8000/v1",
        "the later duplicate value must win"
    );
    assert_eq!(route.upstream_model.as_deref(), Some("SecondModel"));
}

// ---------------------------------------------------------------------------
// `.toml` is read-only: `configure` writes YAML, `.toml` is never written
// ---------------------------------------------------------------------------

/// Writing a config to a `.toml` path is rejected cleanly (not a panic) and
/// never produces a file — `configure` writes YAML, `.toml` is read-only.
#[test]
fn writing_config_to_toml_path_errors_and_creates_no_file() {
    let path = std::env::temp_dir().join(format!(
        "llmconduit-g7-write-{}.toml",
        uuid::Uuid::new_v4().simple()
    ));
    let config = PersistedConfig::default();
    let result = write_persisted_config(&path, &config);

    assert!(
        result.is_err(),
        "writing to a .toml path must be a clean Err"
    );
    let message = result.unwrap_err();
    assert!(
        message.contains("read-only") || message.contains(".toml"),
        "error must explain .toml is read-only: {message}"
    );
    assert!(
        !path.exists(),
        "a rejected .toml write must not leave a file behind"
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// `upstreams:` entries are pure named endpoints
// ---------------------------------------------------------------------------

#[test]
fn upstream_entries_parse_short_and_aliased_names() {
    let yaml = r#"
upstreams:
  - name: local
    url: "http://127.0.0.1:8000/v1"
  - name: aliased
    upstream_base_url: "http://127.0.0.1:8001/v1"
    upstream_api_key: "k"
"#;
    let persisted: PersistedConfig = serde_yaml::from_str(yaml).expect("yaml");
    let config = Config::from_persisted(&persisted).expect("config");
    assert_eq!(config.upstreams[0].name, "local");
    assert_eq!(config.upstreams[1].url.as_str(), "http://127.0.0.1:8001/v1");
    assert_eq!(config.upstreams[1].api_key.as_deref(), Some("k"));
}

#[test]
fn upstream_entry_rejects_removed_keys_and_duplicates() {
    for (yaml_fragment, needle) in [
        ("upstream_model: \"m\"", "upstream_model"),
        ("fallback_upstreams: []", "fallback_upstreams"),
    ] {
        let yaml =
            format!("upstreams:\n  - name: local\n    url: \"http://h/v1\"\n    {yaml_fragment}\n");
        let persisted: PersistedConfig = serde_yaml::from_str(&yaml).expect("yaml");
        let err = Config::from_persisted(&persisted).expect_err("must reject");
        assert!(err.contains(needle) && err.contains("Migrating"), "{err}");
    }
    let dup = "upstreams:\n  - name: local\n    url: \"http://h/v1\"\n  - name: LOCAL\n    url: \"http://h2/v1\"\n";
    let persisted: PersistedConfig = serde_yaml::from_str(dup).expect("yaml");
    assert!(
        Config::from_persisted(&persisted)
            .expect_err("dup")
            .contains("duplicate upstream name")
    );
}

#[test]
fn profile_parses_upstream_fallbacks_and_image_analysis() {
    let yaml = r#"
upstreams:
  - { name: local, url: "http://h/v1" }
  - { name: or, url: "http://o/v1" }
  - { name: vl, url: "http://v/v1" }
model_profiles:
  GLM-5.2:
    upstream: local
    fallbacks:
      - { upstream: or, model: "z-ai/glm-5.2" }
      - vl
    image_analysis: { model: Qwen3-VL }
  Qwen3-VL:
    upstream: vl
"#;
    // Parse-level assertions only: resolve_route lands in Task 5. In this task
    // the runtime ModelProfile carries `upstream: Option<String>` (no
    // requiredness validation yet); Task 5 tightens it to `String` and
    // migrates these assertions to resolve_route.
    let config = Config::from_persisted(&serde_yaml::from_str(yaml).unwrap()).unwrap();
    let profile = config.model_profile("GLM-5.2").expect("profile");
    assert_eq!(profile.upstream.as_deref(), Some("local"));
    assert_eq!(profile.fallbacks[0].model.as_deref(), Some("z-ai/glm-5.2"));
    assert_eq!(profile.fallbacks[1].upstream, "vl");
    assert!(profile.fallbacks[1].model.is_none());
    assert_eq!(profile.image_analysis.as_ref().unwrap().model, "Qwen3-VL");
    // The shorthand-kwargs sweep must not let the new typed keys leak in.
    assert!(profile.upstream_chat_kwargs.is_empty());
}

#[test]
fn profile_native_vision_is_a_startup_error() {
    let yaml = "upstreams: [{ name: l, url: \"http://h/v1\" }]\nmodel_profiles:\n  M: { upstream: l, native_vision: true }\n";
    let err = Config::from_persisted(&serde_yaml::from_str(yaml).unwrap()).unwrap_err();
    assert!(
        err.contains("native_vision") && err.contains("image_analysis"),
        "{err}"
    );
}

#[test]
fn extends_template_supplies_upstream_and_fallbacks_whole_value() {
    let yaml = r#"
upstreams: [{ name: l, url: "http://h/v1" }, { name: o, url: "http://o/v1" }]
model_profile_templates:
  routed: { upstream: l, fallbacks: [o] }
model_profiles:
  A: { extends: [routed] }
  B: { extends: [routed], fallbacks: [] }
"#;
    let config = Config::from_persisted(&serde_yaml::from_str(yaml).unwrap()).unwrap();
    assert_eq!(config.model_profile("A").unwrap().fallbacks.len(), 1);
    assert!(config.model_profile("B").unwrap().fallbacks.is_empty());
}

// ---------------------------------------------------------------------------
// `OrderedModelProfiles`: declaration order, duplicate-key collapse, round
// trip, and the shared glob compiler (Task 3). Resolution against a glob
// profile key lands in Task 5; this task only preserves the persisted shape.
// ---------------------------------------------------------------------------

/// Persisted `model_profiles` preserve DECLARATION order, not alphabetical
/// order. `Zeta` sorts AFTER `Alpha` alphabetically, so a `BTreeMap` would
/// reorder them; the ordered structure keeps file order. Mirrors
/// `overlapping_glob_routes_preserve_declaration_order_not_alphabetical`.
#[test]
fn model_profiles_preserve_declaration_order_not_alphabetical() {
    let zeta_first: PersistedConfig = serde_yaml::from_str(
        r#"
model_profiles:
  Zeta:
    upstream: z
  Alpha:
    upstream: a
"#,
    )
    .expect("yaml");
    let names: Vec<&str> = zeta_first
        .model_profiles
        .0
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["Zeta", "Alpha"],
        "declaration order must be preserved (a BTreeMap would sort Alpha first)"
    );

    // Reversed declaration => reversed resolved order.
    let alpha_first: PersistedConfig = serde_yaml::from_str(
        r#"
model_profiles:
  Alpha:
    upstream: a
  Zeta:
    upstream: z
"#,
    )
    .expect("yaml");
    let names: Vec<&str> = alpha_first
        .model_profiles
        .0
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, vec!["Alpha", "Zeta"]);
}

/// Duplicate profile keys collapse to last-wins (later value replaces the
/// first, preserving the first position), matching `OrderedModelRoutes` and
/// claude-relay dict semantics rather than keeping a shadowed first entry.
#[test]
fn duplicate_model_profile_keys_collapse_to_last_wins() {
    let persisted: PersistedConfig = serde_yaml::from_str(
        r#"
model_profiles:
  GLM-5.2:
    upstream: first
  GLM-5.2:
    upstream: second
"#,
    )
    .expect("yaml");

    assert_eq!(
        persisted.model_profiles.0.len(),
        1,
        "duplicate keys must collapse to a single profile"
    );
    let (name, profile) = &persisted.model_profiles.0[0];
    assert_eq!(name, "GLM-5.2");
    assert_eq!(
        profile.upstream.as_deref(),
        Some("second"),
        "the later duplicate value must win"
    );
}

/// `model_profiles` written by `write_persisted_config` reload through
/// `load_persisted_config` as a MAP, preserving values AND declaration order.
/// Mirrors `model_routes_round_trip_through_write_and_reload_as_map`.
#[test]
fn model_profiles_round_trip_through_write_and_reload_as_map() {
    let config = PersistedConfig {
        // Declared in non-alphabetical order to also lock order across the trip.
        model_profiles: OrderedModelProfiles(vec![
            (
                "Zeta-Model".to_string(),
                PersistedModelProfile {
                    upstream: Some("local".to_string()),
                    ..Default::default()
                },
            ),
            (
                "Alpha-Model".to_string(),
                PersistedModelProfile {
                    upstream: Some("vl".to_string()),
                    ..Default::default()
                },
            ),
        ]),
        ..PersistedConfig::default()
    };

    // The serialized form must be a MAP (`name: profile`), not a sequence.
    let yaml = serde_yaml::to_string(&config).expect("serialize config");
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("parse yaml");
    assert!(
        parsed["model_profiles"].is_mapping(),
        "model_profiles must serialize as a YAML map, not a sequence:\n{yaml}"
    );

    let path = std::env::temp_dir().join(format!(
        "llmconduit-t3-rt-{}.yaml",
        uuid::Uuid::new_v4().simple()
    ));
    write_persisted_config(&path, &config).expect("write config");
    let reloaded = load_persisted_config(&path).expect("reload config");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        reloaded, config,
        "profiles must round-trip through write + reload unchanged"
    );
    let names: Vec<&str> = reloaded
        .model_profiles
        .0
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["Zeta-Model", "Alpha-Model"],
        "declaration order must survive the round trip"
    );
}

/// `compile_model_glob` compiles a case-insensitive matcher for a key holding
/// any of `*?[`.
#[test]
fn compile_model_glob_matches_case_insensitively() {
    let regex = compile_model_glob("claude-*")
        .expect("valid glob")
        .expect("glob pattern must compile a matcher");
    assert!(regex.is_match("Claude-Opus-4"));
}

/// A literal key (no glob metacharacters) compiles to `None`, matched by exact
/// comparison instead of a regex.
#[test]
fn compile_model_glob_returns_none_for_literal_key() {
    assert!(
        compile_model_glob("GLM-5.2")
            .expect("literal key")
            .is_none(),
        "a literal key must not compile a glob matcher"
    );
}
