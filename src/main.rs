//! Token Router — multi-backend token-level LLM router.
//!
//! Sits between any OpenAI-compatible client and multiple inference backends
//! (NPU, GPU, MLX, vLLM), routing individual tokens based on strategy.
//!
//! Usage:
//!   token-router --config router.toml
//!   token-router  # uses default config (passthrough to localhost:13305)

mod backend;
mod config;
mod context;
mod strategy;
mod stream;

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderValue, Method, StatusCode},
    response::Response,
    routing::{get, post},
};
use backend::{BackendClient, BackendPool};
use clap::Parser;
use config::Config;
use context::{Context as RouterContext, Message as RouterMessage};
use std::sync::Arc;
use bytes::Bytes;
use strategy::{CascadeStrategy, ContentRouterStrategy, PassthroughStrategy, SpecDecodeStrategy, RouterStrategy, RoutingDecision};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

/// Token Router CLI.
#[derive(Parser, Debug)]
#[command(name = "token-router", about = "Token-level router for multi-backend LLM inference")]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "router.toml")]
    config: String,
}

/// Shared application state.
struct AppState {
    pool: BackendPool,
    strategy: Box<dyn RouterStrategy>,
    #[allow(dead_code)]
    config: Config,
}

/// Build the routing strategy from config.
fn build_strategy(config: &Config) -> Box<dyn RouterStrategy> {
    let name = &config.server.default_strategy;

    if let Some(strat_cfg) = config.strategies.get(name) {
        match strat_cfg {
            config::StrategyConfig::Passthrough { backend } => {
                Box::new(PassthroughStrategy { backend: backend.clone() })
            }
            config::StrategyConfig::Cascade {
                small_backend,
                large_backend,
                confidence_threshold,
                min_context_for_large,
            } => {
                Box::new(CascadeStrategy {
                    small_backend: small_backend.clone(),
                    large_backend: large_backend.clone(),
                    confidence_threshold: *confidence_threshold,
                    min_context_for_large: *min_context_for_large,
                })
            }
            config::StrategyConfig::ContentRouter {
                small_backend,
                large_backend,
                gpu_keywords,
                max_small_tokens,
            } => {
                Box::new(ContentRouterStrategy {
                    small_backend: small_backend.clone(),
                    large_backend: large_backend.clone(),
                    gpu_keywords: gpu_keywords.clone(),
                    max_small_tokens: *max_small_tokens,
                })
            }
            config::StrategyConfig::SpecDecode {
                draft_backend,
                target_backend,
                n_draft,
                acceptance_threshold,
            } => {
                Box::new(SpecDecodeStrategy {
                    draft_backend: draft_backend.clone(),
                    target_backend: target_backend.clone(),
                    n_draft: *n_draft,
                    acceptance_threshold: *acceptance_threshold,
                })
            }
        }
    } else {
        // Fallback: use first configured backend as passthrough
        let backend = config.backends.keys().next()
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        warn!(strategy = %name, "Strategy not found, using passthrough to {}", backend);
        Box::new(PassthroughStrategy { backend })
    }
}

// ─── HTTP handlers ────────────────────────────────────────────────────

/// GET /v1/models — aggregate models from all backends.
async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let mut all_models = Vec::new();
    for backend_id in state.pool.backend_ids() {
        if let Some(client) = state.pool.client(&backend_id) {
            match client.list_models().await {
                Ok(models) => all_models.extend(models),
                Err(e) => warn!(backend = %backend_id, error = %e, "Failed to list models"),
            }
        }
    }

    let models: Vec<serde_json::Value> = all_models
        .into_iter()
        .map(|id| serde_json::json!({
            "id": id,
            "object": "model",
            "created": 0,
            "owned_by": "token-router"
        }))
        .collect();

    let resp = serde_json::json!({
        "object": "list",
        "data": models,
    });

    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}

/// GET /v1/router — status endpoint showing routing state.
async fn router_status(State(state): State<Arc<AppState>>) -> Response {
    let backends: Vec<serde_json::Value> = state.pool.all_states()
        .into_iter()
        .map(|b| serde_json::json!({
            "id": b.id,
            "healthy": b.healthy,
            "models": b.config.models,
            "type": b.config.backend_type,
        }))
        .collect();

    let resp = serde_json::json!({
        "strategy": state.strategy.name(),
        "backends": backends,
    });

    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}

/// POST /v1/chat/completions — route to the appropriate backend.
async fn chat_completion(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": format!("Invalid JSON: {}", e)}}).to_string(),
                ))
                .unwrap();
        }
    };

    let stream = parsed.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let max_tokens = parsed.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(512) as usize;
    let model_hint = parsed.get("model").and_then(|m| m.as_str()).map(String::from);

    // Extract messages from request body
    let messages = parse_openai_messages(&parsed);

    // Build routing context with messages populated
    let ctx = RouterContext {
        messages,
        session_id: uuid::Uuid::new_v4().to_string(),
        max_tokens,
        model_hint,
        stream,
        generated: Vec::new(),
        total_tokens: 0,
    };

    // Log routing decision for observability
    let route_label = if let Some(model) = &ctx.model_hint {
        format!("model_hint={}", model)
    } else {
        format!("strategy={}", state.strategy.name())
    };
    info!(%route_label, "Routing chat completion");

    // Get routing decision
    let decision = state.strategy.route(&ctx, &state.pool).await;

    let (backend, backend_label) = match &decision {
        RoutingDecision::SingleToken { backend } => (backend.clone(), backend.clone()),
        RoutingDecision::Speculative { draft_backend: _, target_backend, n_draft: _ } => {
            (target_backend.clone(), format!("spec-{}", target_backend))
        }
    };

    let client = match state.pool.client(&backend) {
        Some(c) => c,
        None => {
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": format!("Backend '{}' not available", backend)}}).to_string(),
                ))
                .unwrap();
        }
    };

    info!(backend = %backend, strategy = %state.strategy.name(), "Routed request");
    proxy_request(client, parsed, stream, &backend_label).await
}

/// Proxy a request to a backend, handling both streaming and non-streaming.
async fn proxy_request(client: Arc<BackendClient>, body: serde_json::Value, stream: bool, backend_label: &str) -> Response {
    // Remove routing-specific model names from the body
    // The backend will pick the appropriate model

    if stream {
        match client.chat_completion_stream(body).await {
            Ok(upstream_resp) => {
                // Stream the response directly back
                let status = upstream_resp.status();
                let upstream_body = upstream_resp.bytes_stream();

                let body = Body::from_stream(upstream_body);

                let mut resp = Response::new(body);
                *resp.status_mut() = status;
                resp.headers_mut().insert(
                    "content-type",
                    "text/event-stream".parse().unwrap(),
                );
                resp.headers_mut().insert(
                    "cache-control",
                    "no-cache".parse().unwrap(),
                );
                resp.headers_mut().insert(
                    "x-route-backend",
                    HeaderValue::from_str(backend_label).unwrap(),
                );
                resp
            }
            Err(e) => {
                error!("Streaming proxy error: {}", e);
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"error": {"message": format!("Backend error: {}", e)}}).to_string(),
                    ))
                    .unwrap()
            }
        }
    } else {
        match client.chat_completion(body).await {
            Ok(result) => {
                Response::builder()
                    .header("content-type", "application/json")
                    .header("x-route-backend", backend_label)
                    .body(Body::from(serde_json::to_string(&result).unwrap()))
                    .unwrap()
            }
            Err(e) => {
                error!("Non-streaming proxy error: {}", e);
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"error": {"message": format!("Backend error: {}", e)}}).to_string(),
                    ))
                    .unwrap()
            }
        }
    }
}

/// POST /v1/completions — simple text completion passthrough.
async fn completions(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Response {
    let _parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": format!("Invalid JSON: {}", e)}}).to_string(),
                ))
                .unwrap();
        }
    };

    // For MVP, just proxy to first available backend
    let backend_id = state.pool.backend_ids().first().cloned().unwrap_or_default();
    let client = match state.pool.client(&backend_id) {
        Some(c) => c,
        None => {
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": "No backends available"}}).to_string(),
                ))
                .unwrap();
        }
    };

    match client.completion(_parsed).await {
        Ok(result) => {
            Response::builder()
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&result).unwrap()))
                .unwrap()
        }
        Err(e) => {
            error!("Completion proxy error: {}", e);
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": format!("Backend error: {}", e)}}).to_string(),
                ))
                .unwrap()
        }
    }
}

/// Parse OpenAI chat messages from the request body into our Message format.
/// Handles both string content and content arrays.
fn parse_openai_messages(parsed: &serde_json::Value) -> Vec<RouterMessage> {
    let mut messages = Vec::new();

    if let Some(arr) = parsed["messages"].as_array() {
        for msg in arr {
            let role = msg["role"].as_str().unwrap_or("user").to_string();
            let content = extract_message_content(msg);
            messages.push(RouterMessage { role, content });
        }
    }

    messages
}

/// Extract text content from an OpenAI message (handles string or content array).
fn extract_message_content(msg: &serde_json::Value) -> String {
    match &msg["content"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if part["type"] == "text" {
                    if let Some(t) = part["text"].as_str() {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(t);
                    }
                }
            }
            text
        }
        _ => String::new(),
    }
}

/// Get the concatenated user message text for routing decisions.
#[allow(dead_code)]
fn get_user_text(messages: &[RouterMessage]) -> String {
    messages
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .collect::<Vec<&str>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_messages() {
        let json = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Write a sorting function in Rust"}
            ]
        });
        let msgs = parse_openai_messages(&json);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "You are helpful");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "Write a sorting function in Rust");
    }

    #[test]
    fn test_parse_multimodal_content() {
        let json = serde_json::json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]}
            ]
        });
        let msgs = parse_openai_messages(&json);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "What is in this image?");
    }

    #[test]
    fn test_empty_messages() {
        let json = serde_json::json!({"messages": []});
        let msgs = parse_openai_messages(&json);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn test_no_messages_field() {
        let json = serde_json::json!({"model": "test", "max_tokens": 100});
        let msgs = parse_openai_messages(&json);
        assert_eq!(msgs.len(), 0);
    }
}

// ─── Main ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load config
    let config = if std::path::Path::new(&args.config).exists() {
        Config::from_file(&args.config)?
    } else {
        info!("No config file found, using defaults");
        let mut cfg = Config::default_config();
        cfg.backends.insert(
            "default".to_string(),
            config::BackendConfig {
                backend_type: config::BackendType::OpenAI,
                base_url: "http://127.0.0.1:13305/v1".to_string(),
                api_key: None,
                models: vec!["*".to_string()],
                speed_tok_s: Some(100.0),
                cost_per_token: 1.0,
            },
        );
        cfg
    };

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(&config.server.log_level)
        .init();

    info!("Token Router v{}", env!("CARGO_PKG_VERSION"));
    info!("Strategy: {}", config.server.default_strategy);
    info!("Listen:   {}", config.server.listen);

    // Build backend pool
    let pool = BackendPool::from_config(config.backends.clone());

    // Run initial health checks
    pool.health_check_all().await;

    // Build strategy
    let strategy = build_strategy(&config);

    // Shared state
    let state = Arc::new(AppState {
        pool,
        strategy,
        config: config.clone(),
    });

    // CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/router", get(router_status))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/completions", post(completions))
        .route("/health", get(|| async { "ok" }))
        .layer(cors)
        .with_state(state);

    // Start server
    let listener = TcpListener::bind(&config.server.listen).await?;
    info!("Token Router listening on {}", config.server.listen);

    axum::serve(listener, app).await?;

    Ok(())
}
