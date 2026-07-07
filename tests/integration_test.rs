//! Integration tests — test routing decisions and message parsing.
//!
//! Tests the core routing logic directly (no HTTP server orchestration).

use token_router::config::Config;
use token_router::handlers;

fn test_cascade_config() -> Config {
    Config::from_str(r#"
[server]
listen = "127.0.0.1:0"
default_strategy = "cascade"

[backends.npu]
type = "openai"
base_url = "http://127.0.0.1:1/v1"
models = ["npu"]
speed_tok_s = 291

[backends.gpu]
type = "openai"
base_url = "http://127.0.0.1:2/v1"
models = ["gpu"]
speed_tok_s = 113

[strategies.cascade]
type = "cascade"
small_backend = "npu"
large_backend = "gpu"
confidence_threshold = -1.5
min_context_for_large = 50
"#).unwrap()
}

#[test]
fn test_build_cascade_strategy() {
    let config = test_cascade_config();
    let strat = handlers::build_strategy(&config);
    assert_eq!(strat.name(), "cascade");
}

#[test]
fn test_build_passthrough_strategy() {
    let config = Config::from_str(r#"
[server]
default_strategy = "passthrough"
[backends.gpu]
type = "openai"
base_url = "http://127.0.0.1:1/v1"
models = ["test"]
[strategies.passthrough]
type = "passthrough"
backend = "gpu"
"#).unwrap();
    let strat = handlers::build_strategy(&config);
    assert_eq!(strat.name(), "passthrough");
}

#[test]
fn test_parse_messages_simple() {
    let json = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
    });
    let msgs = handlers::parse_openai_messages(&json);
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "system");
    assert_eq!(msgs[0].content, "You are helpful");
    assert_eq!(msgs[1].role, "user");
    assert_eq!(msgs[1].content, "Hello");
}

#[test]
fn test_parse_messages_multimodal() {
    let json = serde_json::json!({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:img"}}
        ]}]
    });
    let msgs = handlers::parse_openai_messages(&json);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content, "What is in this image?");
}

#[test]
fn test_parse_messages_empty() {
    let json = serde_json::json!({"messages": []});
    let msgs = handlers::parse_openai_messages(&json);
    assert_eq!(msgs.len(), 0);
}

#[test]
fn test_check_logprobs_below_threshold() {
    let response = serde_json::json!({
        "choices": [{
            "logprobs": {
                "content": [
                    {"token": "Hello", "logprob": -0.1, "bytes": null, "top_logprobs": []},
                    {"token": " world", "logprob": -2.5, "bytes": null, "top_logprobs": []}
                ]
            }
        }]
    });
    assert!(handlers::check_logprobs(&response, -1.0),
        "Should detect logprob -2.5 < threshold -1.0");
}

#[test]
fn test_check_logprobs_above_threshold() {
    let response = serde_json::json!({
        "choices": [{
            "logprobs": {
                "content": [
                    {"token": "Hello", "logprob": -0.1, "bytes": null, "top_logprobs": []}
                ]
            }
        }]
    });
    assert!(!handlers::check_logprobs(&response, -1.0),
        "Should NOT flag logprob -0.1 < threshold -1.0");
}

#[test]
fn test_check_logprobs_no_logprobs() {
    let response = serde_json::json!({
        "choices": [{"message": {"content": "Hello"}}]
    });
    assert!(!handlers::check_logprobs(&response, -1.0),
        "Should handle missing logprobs gracefully");
}

#[test]
fn test_validate_config_ok() {
    let config = test_cascade_config();
    let warnings = handlers::validate_config(&config);
    assert!(warnings.is_empty(), "Expected no warnings, got: {:?}", warnings);
}

#[test]
fn test_validate_config_missing_backend() {
    let config = Config::from_str(r#"
[server]
default_strategy = "cascade"
[backends.npu]
type = "openai"
base_url = "http://127.0.0.1:1/v1"
models = ["npu"]
[strategies.cascade]
type = "cascade"
small_backend = "npu"
large_backend = "nonexistent"
"#).unwrap();
    let warnings = handlers::validate_config(&config);
    assert!(!warnings.is_empty(), "Expected warning about missing backend");
    assert!(warnings[0].contains("nonexistent"));
}
