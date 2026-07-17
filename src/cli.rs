use crate::config::OrderedModelProfiles;
use crate::config::PersistedConfig;
use crate::config::PersistedModelProfile;
use crate::config::PersistedUpstream;
use crate::config::default_config_path;
use crate::config::load_persisted_config;
use crate::config::write_persisted_config;
use clap::Parser;
use clap::Subcommand;
use dialoguer::Confirm;
use dialoguer::Input;
use dialoguer::Password;
use dialoguer::theme::ColorfulTheme;
use serde_json::Map as JsonMap;
use serde_json::Value as JsonValue;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "llmconduit",
    version = crate::VERSION,
    about = "LLM API gateway for translating, normalizing, and extending model traffic"
)]
pub struct Cli {
    /// Enable the embedded request debug UI at /debug.
    #[arg(long, global = true, default_value_t = false)]
    pub with_debug_ui: bool,
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Start the gateway server.
    Start {
        /// Path to the config file. Defaults to ~/.config/llmconduit/config.yaml
        #[arg(long)]
        config: Option<PathBuf>,
        /// Dump raw model delta text to the terminal while the gateway is running.
        #[arg(long, default_value_t = false)]
        raw: bool,
    },
    /// Run the interactive configuration flow and write a config file.
    Configure {
        /// Path to the config file. Defaults to ~/.config/llmconduit/config.yaml
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Diff consecutive upstream request log entries and highlight unstable prefixes.
    AnalyzeLog {
        /// Path to the config file. Defaults to ~/.config/llmconduit/config.yaml
        #[arg(long)]
        config: Option<PathBuf>,
        /// Path to the JSONL request log. Defaults to upstream_request_log_path from config.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Maximum number of consecutive pairs to report.
        #[arg(long, default_value_t = 10)]
        pairs: usize,
    },
}

pub fn resolve_config_path(path: Option<PathBuf>) -> Result<PathBuf, String> {
    path.map(Ok).unwrap_or_else(default_config_path)
}

/// Values gathered from the interactive prompts in [`run_configure_flow`],
/// separated out so [`build_configured_persisted_config`] can be exercised
/// without a TTY.
struct ConfigureFlowInputs {
    bind_addr: String,
    upstream_base_url: String,
    upstream_api_key: Option<String>,
    /// Raw prompt text; blank means "pass the request model through" (no
    /// `upstream_model` override on the `"*"` profile).
    upstream_model: String,
    /// Raw prompt text; blank means logging is disabled.
    upstream_request_log_path: String,
    upstream_chat_kwargs: JsonMap<String, JsonValue>,
    brave_base_url: String,
    /// Raw prompt text; blank means no key.
    brave_api_key: String,
    brave_max_results: usize,
    request_timeout_secs: u64,
}

/// Builds the emitted config from prompt inputs plus whatever wasn't prompted
/// for: a single `default` upstream and a `"*"` catch-all profile pointing at
/// it, with `upstream_model` set only when a default model was entered (so a
/// blank entry keeps passing the request model through).
fn build_configured_persisted_config(
    existing: &PersistedConfig,
    inputs: ConfigureFlowInputs,
) -> PersistedConfig {
    let default_upstream = PersistedUpstream {
        name: "default".to_string(),
        url: inputs.upstream_base_url,
        api_key: inputs.upstream_api_key,
        chat_kwargs: inputs.upstream_chat_kwargs,
        request_log_path: (!inputs.upstream_request_log_path.trim().is_empty())
            .then(|| inputs.upstream_request_log_path.trim().to_string()),
        upstream_model: None,
        fallback_upstreams: None,
    };
    let default_profile = PersistedModelProfile {
        upstream: Some("default".to_string()),
        upstream_model: (!inputs.upstream_model.trim().is_empty())
            .then(|| inputs.upstream_model.trim().to_string()),
        ..PersistedModelProfile::default()
    };

    PersistedConfig {
        bind_addr: inputs.bind_addr,
        upstream_base_url: None,
        upstream_api_key: None,
        upstream_model: None,
        system_prompt_prefix: existing.system_prompt_prefix.clone(),
        upstream_request_log_path: None,
        // F1: not interactively prompted (an advanced, opt-in knob, like
        // `debug_log_max_age_hours` below) -- just carried through unchanged.
        turn_capture_dir: existing.turn_capture_dir.clone(),
        upstream_chat_kwargs: existing.upstream_chat_kwargs.clone(),
        upstreams: vec![default_upstream],
        fallback_upstreams: None,
        upstream_failure_cooldown_secs: existing.upstream_failure_cooldown_secs,
        model_profile_templates: existing.model_profile_templates.clone(),
        model_profiles: OrderedModelProfiles(vec![("*".to_string(), default_profile)]),
        model_routes: None,
        template_family: existing.template_family.clone(),
        brave_base_url: inputs.brave_base_url,
        brave_api_key: (!inputs.brave_api_key.trim().is_empty()).then_some(inputs.brave_api_key),
        brave_max_results: inputs.brave_max_results,
        request_timeout_secs: inputs.request_timeout_secs,
        connect_timeout_secs: existing.connect_timeout_secs,
        max_web_search_rounds: existing.max_web_search_rounds,
        flatten_content: existing.flatten_content,
        max_replay_entries: existing.max_replay_entries,
        debug_log_max_age_hours: existing.debug_log_max_age_hours,
        min_completion_tokens: existing.min_completion_tokens,
        max_sse_frame_bytes: existing.max_sse_frame_bytes,
        max_request_body_bytes: existing.max_request_body_bytes,
        image_agent_enabled: None,
        vision_url: None,
        vision_model: None,
        image_cache_max_size: existing.image_cache_max_size,
        image_cache_ttl_secs: existing.image_cache_ttl_secs,
        unsupported_image_policy: None,
        price_table: existing.price_table.clone(),
    }
}

/// Warns before the wizard flattens an existing config: `build_configured_persisted_config`
/// always emits exactly one upstream and one profile, so a hand-edited config
/// with more than one of either would otherwise lose the rest silently at
/// write time.
fn configure_collapse_warning(existing: &PersistedConfig) -> Option<String> {
    let mut dropped = Vec::new();

    if existing.upstreams.len() > 1 {
        let names = existing
            .upstreams
            .iter()
            .map(|upstream| upstream.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        dropped.push(format!("{} upstreams ({names})", existing.upstreams.len()));
    }

    if existing.model_profiles.0.len() > 1 {
        let keys = existing
            .model_profiles
            .0
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        dropped.push(format!(
            "{} profiles ({keys})",
            existing.model_profiles.0.len()
        ));
    }

    if dropped.is_empty() {
        return None;
    }

    Some(format!(
        "The existing config has {} that will be collapsed into a single `default` upstream and \
         a `\"*\"` profile if you continue.",
        dropped.join(" and ")
    ))
}

pub fn run_configure_flow(path: PathBuf) -> Result<PersistedConfig, String> {
    let existing = load_persisted_config(&path)?;
    let theme = ColorfulTheme::default();

    println!("Configuring llmconduit");
    println!("Config file: {}", path.display());

    if let Some(warning) = configure_collapse_warning(&existing) {
        println!("{warning}");
        let proceed = Confirm::with_theme(&theme)
            .with_prompt("Continue anyway?")
            .default(false)
            .interact()
            .map_err(|err| format!("failed to confirm config collapse: {err}"))?;
        if !proceed {
            return Err("configuration cancelled".to_string());
        }
    }

    let bind_addr = Input::with_theme(&theme)
        .with_prompt("Bind address")
        .default(existing.bind_addr.clone())
        .interact_text()
        .map_err(|err| format!("failed to read bind address: {err}"))?;

    // The single default endpoint is persisted as one `upstreams` entry named
    // "default" plus a `"*"` catch-all profile pointing at it, so the interactive
    // prompts pre-fill from (and write back) that shape.
    let existing_default = existing
        .upstreams
        .iter()
        .find(|entry| entry.name == "default")
        .or_else(|| existing.upstreams.first());
    let existing_url = existing_default
        .map(|entry| entry.url.clone())
        .unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_string());
    let existing_api_key = existing_default.and_then(|entry| entry.api_key.clone());
    let existing_entry_chat_kwargs = existing_default
        .map(|entry| entry.chat_kwargs.clone())
        .unwrap_or_default();
    let existing_request_log_path = existing_default
        .and_then(|entry| entry.request_log_path.clone())
        .or_else(|| existing.upstream_request_log_path.clone());
    let existing_default_model = existing
        .model_profiles
        .0
        .iter()
        .find(|(name, _)| name == "*")
        .and_then(|(_, profile)| profile.upstream_model.clone());

    let upstream_base_url = Input::with_theme(&theme)
        .with_prompt("Upstream chat-completions base URL")
        .default(existing_url)
        .interact_text()
        .map_err(|err| format!("failed to read upstream URL: {err}"))?;
    let upstream_api_key = match existing_api_key {
        Some(existing_api_key) => {
            let keep_existing = Confirm::with_theme(&theme)
                .with_prompt("Keep existing upstream API key?")
                .default(true)
                .interact()
                .map_err(|err| format!("failed to confirm upstream API key: {err}"))?;
            if keep_existing {
                Some(existing_api_key)
            } else {
                let value = Password::with_theme(&theme)
                    .with_prompt("Upstream API key (leave blank for local/no auth)")
                    .allow_empty_password(true)
                    .interact()
                    .map_err(|err| format!("failed to read upstream API key: {err}"))?;
                (!value.trim().is_empty()).then_some(value)
            }
        }
        None => {
            let value = Password::with_theme(&theme)
                .with_prompt("Upstream API key (leave blank for local/no auth)")
                .allow_empty_password(true)
                .interact()
                .map_err(|err| format!("failed to read upstream API key: {err}"))?;
            (!value.trim().is_empty()).then_some(value)
        }
    };
    let upstream_model = Input::with_theme(&theme)
        .with_prompt("Upstream model override (leave blank to pass through request model)")
        .allow_empty(true)
        .default(existing_default_model.unwrap_or_default())
        .interact_text()
        .map_err(|err| format!("failed to read upstream model override: {err}"))?;
    let upstream_request_log_path = Input::with_theme(&theme)
        .with_prompt("Upstream request JSONL log path (leave blank to disable)")
        .allow_empty(true)
        .default(existing_request_log_path.unwrap_or_default())
        .interact_text()
        .map_err(|err| format!("failed to read upstream request log path: {err}"))?;
    let upstream_chat_kwargs = Input::with_theme(&theme)
        .with_prompt("Extra upstream chat kwargs as JSON object (leave blank for none)")
        .allow_empty(true)
        .default(if existing_entry_chat_kwargs.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&existing_entry_chat_kwargs)
                .map_err(|err| format!("failed to encode upstream chat kwargs: {err}"))?
        })
        .interact_text()
        .map_err(|err| format!("failed to read upstream chat kwargs: {err}"))?;
    let brave_base_url = Input::with_theme(&theme)
        .with_prompt("Brave Search base URL")
        .default(existing.brave_base_url.clone())
        .interact_text()
        .map_err(|err| format!("failed to read Brave URL: {err}"))?;
    let brave_api_key = Password::with_theme(&theme)
        .with_prompt("Brave Search API key (leave blank to disable provider-side web_search)")
        .allow_empty_password(true)
        .interact()
        .map_err(|err| format!("failed to read Brave API key: {err}"))?;
    let brave_max_results = Input::with_theme(&theme)
        .with_prompt("Brave max results")
        .default(existing.brave_max_results)
        .interact_text()
        .map_err(|err| format!("failed to read Brave max results: {err}"))?;
    let request_timeout_secs = Input::with_theme(&theme)
        .with_prompt("Request timeout (seconds)")
        .default(existing.request_timeout_secs)
        .interact_text()
        .map_err(|err| format!("failed to read timeout: {err}"))?;

    let upstream_chat_kwargs = if upstream_chat_kwargs.trim().is_empty() {
        JsonMap::new()
    } else {
        serde_json::from_str::<JsonMap<String, JsonValue>>(&upstream_chat_kwargs)
            .map_err(|err| format!("invalid upstream chat kwargs JSON: {err}"))?
    };

    let config = build_configured_persisted_config(
        &existing,
        ConfigureFlowInputs {
            bind_addr,
            upstream_base_url,
            upstream_api_key,
            upstream_model,
            upstream_request_log_path,
            upstream_chat_kwargs,
            brave_base_url,
            brave_api_key,
            brave_max_results,
            request_timeout_secs,
        },
    );

    let should_write = Confirm::with_theme(&theme)
        .with_prompt(format!("Write configuration to {}?", path.display()))
        .default(true)
        .interact()
        .map_err(|err| format!("failed to confirm config write: {err}"))?;
    if !should_write {
        return Err("configuration cancelled".to_string());
    }

    write_persisted_config(&path, &config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn base_inputs() -> ConfigureFlowInputs {
        ConfigureFlowInputs {
            bind_addr: "127.0.0.1:4000".to_string(),
            upstream_base_url: "http://127.0.0.1:8000/v1".to_string(),
            upstream_api_key: Some("secret".to_string()),
            upstream_model: String::new(),
            upstream_request_log_path: String::new(),
            upstream_chat_kwargs: JsonMap::new(),
            brave_base_url: "https://api.search.brave.com/res/v1".to_string(),
            brave_api_key: String::new(),
            brave_max_results: 5,
            request_timeout_secs: 60,
        }
    }

    /// A blank "default model" prompt must keep passing the request model
    /// through, since that's the only way an unset upstream model still works.
    #[test]
    fn wizard_output_without_default_model_passes_request_model_through() {
        let persisted =
            build_configured_persisted_config(&PersistedConfig::default(), base_inputs());

        let config = Config::from_persisted(&persisted).expect("valid config");
        let route = config.resolve_route("anything").expect("route resolves");
        assert_eq!(route.profile.upstream, "default");
        assert_eq!(route.served_model, "anything");
    }

    /// An entered default model becomes the profile's `upstream_model`, so
    /// every request routes to it regardless of the request's own model field.
    #[test]
    fn wizard_output_with_default_model_serves_entered_model() {
        let mut inputs = base_inputs();
        inputs.upstream_model = "my-model".to_string();
        let persisted = build_configured_persisted_config(&PersistedConfig::default(), inputs);

        let config = Config::from_persisted(&persisted).expect("valid config");
        let route = config.resolve_route("anything").expect("route resolves");
        assert_eq!(route.profile.upstream, "default");
        assert_eq!(route.served_model, "my-model");
    }

    #[test]
    fn collapse_warning_is_none_for_single_upstream_and_profile() {
        let existing = PersistedConfig::default();

        assert_eq!(configure_collapse_warning(&existing), None);
    }

    #[test]
    fn collapse_warning_names_extra_upstream() {
        let mut existing = PersistedConfig::default();
        existing.upstreams.push(PersistedUpstream {
            name: "secondary".to_string(),
            url: "http://127.0.0.1:9000/v1".to_string(),
            api_key: None,
            chat_kwargs: JsonMap::new(),
            request_log_path: None,
            upstream_model: None,
            fallback_upstreams: None,
        });

        let warning = configure_collapse_warning(&existing).expect("warns about extra upstream");
        assert!(
            warning.contains("secondary"),
            "expected warning to name the extra upstream, got: {warning}"
        );
    }

    #[test]
    fn collapse_warning_names_extra_profile_keys() {
        let mut existing = PersistedConfig::default();
        existing.model_profiles.0.push((
            "coder".to_string(),
            PersistedModelProfile {
                upstream: Some("default".to_string()),
                ..PersistedModelProfile::default()
            },
        ));

        let warning = configure_collapse_warning(&existing).expect("warns about extra profile");
        assert!(
            warning.contains("coder"),
            "expected warning to name the extra profile key, got: {warning}"
        );
    }
}
