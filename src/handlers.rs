//! HTTP handlers and server setup for the Token Router.
//!
//! All handler functions, AppState, metrics, and the server run loop live here.

use crate::backend::{BackendClient, BackendPool};
use crate::cascade;
use crate::config::{self, Config};
use crate::context::{Context as RouterContext, Message as RouterMessage};
use crate::strategy::{
    CascadeStrategy, ContentRouterStrategy, PassthroughStrategy,
    RouterStrategy, RoutingDecision, SpecDecodeStrategy,
};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderValue, Method, StatusCode},
    response::Response,
    routing::{get, post},
};
use clap::Parser;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

/// Token Router CLI.
#[derive(Parser, Debug)]
#[command(name = "token-router", about = "Token-level router for multi-backend LLM inference")]
pub struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "router.toml")]
    pub config: String,
}

/// Shared application state.
pub struct AppState {
    pool: Arc<BackendPool>,
    strategy: Box<dyn RouterStrategy>,
    #[allow(dead_code)]
    config: Config,
    metrics: AppMetrics,
}

/// Runtime metrics for observability.
pub struct AppMetrics {
    start_time: Instant,
    requests_total: std::sync::atomic::AtomicU64,
    cascade_switches: std::sync::atomic::AtomicU64,
}

impl AppMetrics {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            requests_total: std::sync::atomic::AtomicU64::new(0),
            cascade_switches: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn inc_requests(&self) {
        self.requests_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_cascade_switches(&self) {
        self.cascade_switches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Build the routing strategy from config.
pub fn build_strategy(config: &Config) -> Box<dyn RouterStrategy> {
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
pub async fn list_models(State(state): State<Arc<AppState>>) -> Response {
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
pub async fn router_status(State(state): State<Arc<AppState>>) -> Response {
    state.pool.refresh_metrics().await;
    let backends: Vec<serde_json::Value> = state.pool.all_states()
        .into_iter()
        .map(|b| serde_json::json!({
            "id": b.id,
            "healthy": b.healthy,
            "type": b.config.backend_type,
            "models": b.config.models,
            "latency_ema_ms": format!("{:.1}", b.latency_ema_ms),
            "circuit_open": b.circuit_open,
            "active_requests": b.active_requests,
            "max_concurrent": b.max_concurrent,
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

/// GET /v1/router/metrics — runtime metrics.
pub async fn router_metrics(State(state): State<Arc<AppState>>) -> Response {
    // Refresh latency metrics from live clients
    state.pool.refresh_metrics().await;

    let uptime = state.metrics.start_time.elapsed().as_secs();
    let total = state.metrics.requests_total.load(std::sync::atomic::Ordering::Relaxed);
    let switches = state.metrics.cascade_switches.load(std::sync::atomic::Ordering::Relaxed);

    let backends: Vec<serde_json::Value> = state.pool.all_states()
        .into_iter()
        .map(|b| serde_json::json!({
            "id": b.id,
            "healthy": b.healthy,
            "type": b.config.backend_type,
            "latency_ema_ms": format!("{:.1}", b.latency_ema_ms),
            "circuit_open": b.circuit_open,
            "active_requests": b.active_requests,
            "max_concurrent": b.max_concurrent,
        }))
        .collect();

    let resp = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "strategy": state.strategy.name(),
        "uptime_seconds": uptime,
        "requests_total": total,
        "cascade_switches": switches,
        "backends": backends,
    });

    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}

/// Validate config at startup — check that strategies reference valid backends.
pub fn validate_config(config: &Config) -> Vec<String> {
    let mut warnings = Vec::new();
    let backend_ids: Vec<&str> = config.backends.keys().map(|s| s.as_str()).collect();

    for (name, strategy) in &config.strategies {
        match strategy {
            config::StrategyConfig::Passthrough { backend } => {
                if !backend_ids.contains(&backend.as_str()) {
                    warnings.push(format!(
                        "Strategy '{}' references unknown backend '{}'", name, backend
                    ));
                }
            }
            config::StrategyConfig::Cascade { small_backend, large_backend, .. } => {
                if !backend_ids.contains(&small_backend.as_str()) {
                    warnings.push(format!(
                        "Cascade strategy '{}' references unknown small_backend '{}'", name, small_backend
                    ));
                }
                if !backend_ids.contains(&large_backend.as_str()) {
                    warnings.push(format!(
                        "Cascade strategy '{}' references unknown large_backend '{}'", name, large_backend
                    ));
                }
            }
            config::StrategyConfig::ContentRouter { small_backend, large_backend, .. } => {
                if !backend_ids.contains(&small_backend.as_str()) {
                    warnings.push(format!(
                        "ContentRouter '{}' references unknown small_backend '{}'", name, small_backend
                    ));
                }
                if !backend_ids.contains(&large_backend.as_str()) {
                    warnings.push(format!(
                        "ContentRouter '{}' references unknown large_backend '{}'", name, large_backend
                    ));
                }
            }
            config::StrategyConfig::SpecDecode { draft_backend, target_backend, .. } => {
                if !backend_ids.contains(&draft_backend.as_str()) {
                    warnings.push(format!(
                        "SpecDecode '{}' references unknown draft_backend '{}'", name, draft_backend
                    ));
                }
                if !backend_ids.contains(&target_backend.as_str()) {
                    warnings.push(format!(
                        "SpecDecode '{}' references unknown target_backend '{}'", name, target_backend
                    ));
                }
            }
        }
    }

    warnings
}

/// POST /v1/chat/completions — route to the appropriate backend.
pub async fn chat_completion(
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

    state.metrics.inc_requests();

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

    // ── Cascade strategy (streaming) — the novel path ────────────────
    // Token-level confidence routing: start NPU, switch to GPU on low confidence.
    if state.strategy.name() == "cascade" && stream {
        // Extract cascade config (backend IDs and threshold)
        let cascade_cfg = match state.config.strategies.get("cascade") {
            Some(config::StrategyConfig::Cascade {
                small_backend,
                large_backend,
                confidence_threshold,
                ..
            }) => (small_backend.clone(), large_backend.clone(), *confidence_threshold),
            _ => {
                // Fallback: use default backends
                let ids = state.pool.backend_ids();
                let npu = ids.iter().find(|id| id.contains("npu")).cloned()
                    .unwrap_or_else(|| ids.first().cloned().unwrap_or_default());
                let gpu = ids.iter().find(|id| id.contains("gpu")).cloned()
                    .unwrap_or_else(|| ids.last().cloned().unwrap_or_default());
                (npu, gpu, -2.5)
            }
        };

        let (npu_id, gpu_id, threshold) = cascade_cfg;

        let npu_client = match state.pool.client(&npu_id) {
            Some(c) => c,
            None => return proxy_single(state, parsed, &gpu_id, stream).await,
        };
        let gpu_client = match state.pool.client(&gpu_id) {
            Some(c) => c,
            None => return proxy_single(state, parsed, &npu_id, stream).await,
        };

        info!(npu = %npu_id, gpu = %gpu_id, threshold = %threshold, "Cascade streaming");

        let stream_body = cascade::cascade_stream(npu_client, gpu_client, parsed, threshold);
        let mut resp = Response::new(Body::from_stream(stream_body));
        resp.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        resp.headers_mut().insert(
            "cache-control",
            HeaderValue::from_static("no-cache"),
        );
        resp.headers_mut().insert(
            "x-route-backend",
            HeaderValue::from_static("cascade"),
        );
        return resp;
    }

    // ── Cascade strategy (non-streaming) ─────────────────────────────
    if state.strategy.name() == "cascade" && !stream {
        return cascade_nonstreaming(state, parsed).await;
    }

    // ── Standard routing: single backend ─────────────────────────────
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

/// Quick proxy to a single backend by ID (used as fallback).
pub async fn proxy_single(
    state: Arc<AppState>,
    body: serde_json::Value,
    backend_id: &str,
    stream: bool,
) -> Response {
    let client = match state.pool.client(backend_id) {
        Some(c) => c,
        None => {
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": {"message": format!("Backend '{}' not available", backend_id)}}).to_string(),
                ))
                .unwrap();
        }
    };
    proxy_request(client, body, stream, backend_id).await
}

/// Cascade routing for non-streaming requests.
/// Sends to NPU with logprobs, checks per-token confidence, re-routes to GPU if needed.
pub async fn cascade_nonstreaming(state: Arc<AppState>, body: serde_json::Value) -> Response {
    let cascade_cfg = match state.config.strategies.get("cascade") {
        Some(config::StrategyConfig::Cascade {
            small_backend,
            large_backend,
            confidence_threshold,
            ..
        }) => (small_backend.clone(), large_backend.clone(), *confidence_threshold),
        _ => return proxy_single(state, body, "gpu", false).await,
    };

    let (npu_id, gpu_id, threshold) = cascade_cfg;

    // Try NPU first with logprobs
    let npu_client = match state.pool.client(&npu_id) {
        Some(c) => c,
        None => return proxy_single(state, body, &gpu_id, false).await,
    };

    let mut npu_body = body.clone();
    npu_body["logprobs"] = serde_json::Value::Bool(true);
    npu_body["top_logprobs"] = serde_json::Value::Number(serde_json::Number::from(1));
    npu_body["stream"] = serde_json::Value::Bool(false);

    match npu_client.chat_completion(npu_body).await {
        Ok(npu_result) => {
            // Check per-token log-probs
            let low_conf = check_logprobs(&npu_result, threshold);

            if low_conf {
                info!("Cascade (non-streaming): low confidence on NPU, rerouting to GPU");
                // Re-send to GPU
                let gpu_client = match state.pool.client(&gpu_id) {
                    Some(c) => c,
                    None => {
                        return Response::builder()
                            .header("content-type", "application/json")
                            .body(Body::from(serde_json::to_string(&npu_result).unwrap()))
                            .unwrap();
                    }
                };

                match gpu_client.chat_completion(body).await {
                    Ok(gpu_result) => {
                        Response::builder()
                            .header("content-type", "application/json")
                            .header("x-route-backend", &gpu_id)
                            .body(Body::from(serde_json::to_string(&gpu_result).unwrap()))
                            .unwrap()
                    }
                    Err(e) => {
                        warn!("GPU fallback failed: {}. Returning NPU result.", e);
                        Response::builder()
                            .header("content-type", "application/json")
                            .header("x-route-backend", &npu_id)
                            .body(Body::from(serde_json::to_string(&npu_result).unwrap()))
                            .unwrap()
                    }
                }
            } else {
                info!("Cascade (non-streaming): NPU confidence OK");
                Response::builder()
                    .header("content-type", "application/json")
                    .header("x-route-backend", &npu_id)
                    .body(Body::from(serde_json::to_string(&npu_result).unwrap()))
                    .unwrap()
            }
        }
        Err(e) => {
            warn!("NPU request failed in cascade: {}. Falling back to GPU.", e);
            proxy_single(state, body, &gpu_id, false).await
        }
    }
}

/// Check if any token in the response has log-prob below threshold.
pub fn check_logprobs(response: &serde_json::Value, threshold: f64) -> bool {
    if let Some(choices) = response["choices"].as_array() {
        for choice in choices {
            if let Some(logprobs) = choice.get("logprobs") {
                if let Some(content) = logprobs["content"].as_array() {
                    for token_info in content {
                        if let Some(lp) = token_info["logprob"].as_f64() {
                            if lp < threshold {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Proxy a request to a backend, handling both streaming and non-streaming.
pub async fn proxy_request(client: Arc<BackendClient>, body: serde_json::Value, stream: bool, backend_label: &str) -> Response {
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
pub async fn completions(
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
pub fn parse_openai_messages(parsed: &serde_json::Value) -> Vec<RouterMessage> {
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
pub fn extract_message_content(msg: &serde_json::Value) -> String {
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

/// Build the axum Router from shared state.
pub fn build_app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/router", get(router_status))
        .route("/v1/router/metrics", get(router_metrics))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/completions", post(completions))
        .route("/health", get(|| async { "ok" }))
        .route("/", get(|| async { "Token Router — see /v1/router for status\n" }))
        .layer(cors)
        .with_state(state)
}

/// Convenience: build app from components (for tests).
pub fn build_app_from_components(pool: BackendPool, strategy: Box<dyn RouterStrategy>, config: Config) -> Router {
    let state = Arc::new(AppState {
        pool: Arc::new(pool),
        strategy,
        config,
        metrics: AppMetrics::new(),
    });
    build_app(state)
}

/// Run the token router server from a parsed Config.
pub async fn run_server(config: Config) -> anyhow::Result<()> {

    // Initialize tracing (non-fatal if already set)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(&config.server.log_level)
        .try_init();

    info!("Token Router v{}", env!("CARGO_PKG_VERSION"));
    info!("Strategy: {}", config.server.default_strategy);
    info!("Listen:   {}", config.server.listen);

    // Validate config
    let warnings = validate_config(&config);
    for w in &warnings {
        warn!("Config warning: {}", w);
    }

    // Build backend pool
    let pool = BackendPool::from_config(config.backends.clone());

    // Run initial health checks
    pool.health_check_all().await;

    // Build strategy
    let strategy = build_strategy(&config);

    let pool = Arc::new(pool);

    // Shared state with metrics
    let state = Arc::new(AppState {
        pool: pool.clone(),
        strategy,
        config: config.clone(),
        metrics: AppMetrics::new(),
    });

    let app = build_app(state.clone());

    // Start server with graceful shutdown
    let listener = TcpListener::bind(&config.server.listen).await?;
    info!("Token Router listening on {}", config.server.listen);

    // Spawn background health check loop (every 30s)
    let pool_for_health = pool.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            pool_for_health.health_check_all().await;
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Token Router shutting down");
    Ok(())
}

/// Wait for SIGTERM or SIGINT to trigger graceful shutdown.
async fn shutdown_signal() {
    use tokio::signal;
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { info!("Received Ctrl+C"); },
        _ = terminate => { info!("Received SIGTERM"); },
    }
}
