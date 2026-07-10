//! HTTP client for OpenAI-compatible inference backends.
//!
//! Wraps `reqwest` with backend-agnostic request/response handling.
//! Supports streaming (SSE) and non-streaming modes.
//!
//! Includes circuit breaker: after N consecutive failures, the circuit
//! opens for a cooldown period before allowing retries (exponential backoff).

use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::warn;

/// Streaming chunk from an SSE response.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub data: String,
}

/// Circuit breaker state for a backend.
#[derive(Debug)]
struct CircuitBreaker {
    consecutive_failures: AtomicU64,
    last_failure_time: RwLock<Option<Instant>>,
    is_open: AtomicBool,
    failure_threshold: u64,
    cooldown_secs: u64,
}

impl CircuitBreaker {
    fn new(failure_threshold: u64, cooldown_secs: u64) -> Self {
        Self {
            consecutive_failures: AtomicU64::new(0),
            last_failure_time: RwLock::new(None),
            is_open: AtomicBool::new(false),
            failure_threshold,
            cooldown_secs,
        }
    }

    /// Check if the circuit allows a request through.
    fn allow_request(&self) -> bool {
        if !self.is_open.load(Ordering::Relaxed) {
            return true;
        }
        // Circuit is open — check if cooldown has elapsed
        if let Some(last) = *self.last_failure_time.blocking_read() {
            if last.elapsed().as_secs() >= self.cooldown_secs {
                // Half-open: allow one probe request
                self.is_open.store(false, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Record a successful request.
    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.is_open.store(false, Ordering::Relaxed);
    }

    /// Record a failed request. Opens circuit if threshold exceeded.
    fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.blocking_write() = Some(Instant::now());
        if failures >= self.failure_threshold {
            warn!(
                "Circuit breaker OPEN after {} consecutive failures (cooldown: {}s)",
                failures, self.cooldown_secs
            );
            self.is_open.store(true, Ordering::Relaxed);
        }
    }
}

/// Latency tracker for a backend (exponential moving average).
#[derive(Debug)]
pub struct LatencyTracker {
    /// EMA of request latency in milliseconds
    ema_ms: RwLock<f64>,
    /// Smoothing factor (0.0-1.0, higher = more weight on recent)
    alpha: f64,
}

impl LatencyTracker {
    pub fn new(initial_ms: f64) -> Self {
        Self {
            ema_ms: RwLock::new(initial_ms),
            alpha: 0.3,
        }
    }

    pub async fn record(&self, latency_ms: f64) {
        let mut ema = self.ema_ms.write().await;
        *ema = self.alpha * latency_ms + (1.0 - self.alpha) * *ema;
    }

    pub async fn current(&self) -> f64 {
        *self.ema_ms.read().await
    }
}

/// HTTP client wrapping a single backend URL.
#[derive(Debug, Clone)]
pub struct BackendClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    circuit: Arc<CircuitBreaker>,
    pub latency: Arc<LatencyTracker>,
}

impl BackendClient {
    /// Create a new client pointing at a backend.
    pub fn new(base_url: &str, api_key: Option<&str>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("Failed to build reqwest client");

        // Normalize: strip trailing /v1 if present (we append our own paths)
        let base_url = base_url.trim_end_matches('/');
        let base_url = base_url
            .strip_suffix("/v1")
            .unwrap_or(base_url)
            .to_string();

        Self {
            client,
            base_url,
            api_key: api_key.map(String::from),
            circuit: Arc::new(CircuitBreaker::new(5, 30)),
            latency: Arc::new(LatencyTracker::new(50.0)),
        }
    }

    /// Get the backend's base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Check if the circuit breaker allows a request.
    #[allow(dead_code)]
    pub fn is_available(&self) -> bool {
        self.circuit.allow_request()
    }

    /// Make an HTTP request with circuit breaker and latency tracking.
    async fn request_with_circuit<F, T>(
        &self,
        operation: &str,
        f: F,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        if !self.circuit.allow_request() {
            anyhow::bail!("Circuit breaker open for backend {}", self.base_url);
        }
        let start = Instant::now();
        match f.await {
            Ok(result) => {
                self.circuit.record_success();
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                self.latency.record(latency_ms).await;
                Ok(result)
            }
            Err(e) => {
                self.circuit.record_failure();
                warn!("Backend {} {} failed: {}", self.base_url, operation, e);
                Err(e)
            }
        }
    }

    /// Health check — GET /v1/models or /
    pub async fn health_check(&self) -> Result<Value> {
        let url = format!("{}/v1/models", self.base_url);
        let mut req = self.client.get(&url).timeout(Duration::from_secs(5));
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        let resp = req.send().await.context("Health check request failed")?;
        let body: Value = resp.json().await.context("Failed to parse health check response")?;
        Ok(body)
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.base_url);
        let mut req = self.client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        let resp = req.send().await.context("Models request failed")?;
        let body: Value = resp.json().await.context("Failed to parse models response")?;

        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    /// Chat completion (non-streaming).
    pub async fn chat_completion(&self, body: Value) -> Result<Value> {
        self.request_with_circuit("chat_completion", async {
            let url = format!("{}/v1/chat/completions", self.base_url);
            let mut req = self
                .client
                .post(&url)
                .json(&body)
                .timeout(Duration::from_secs(600));
            if let Some(key) = &self.api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req.send().await.context("Chat completion request failed")?;
            let result: Value = resp.json().await.context("Failed to parse chat completion response")?;
            Ok(result)
        }).await
    }

    /// Make a POST request with JSON body to an arbitrary endpoint path.
    /// The path should start with "/" and will be appended to base_url.
    /// Useful for KV cache handoff endpoints: /v1/kv_cache/export, /v1/kv_cache/import.
    pub async fn post_json(&self, path: &str, body: Value) -> Result<Value> {
        self.request_with_circuit("post_json", async {
            let url = format!("{}{}", self.base_url, path);
            let mut req = self
                .client
                .post(&url)
                .json(&body)
                .timeout(std::time::Duration::from_secs(600));
            if let Some(key) = &self.api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req.send().await.context("POST request failed")?;
            let result: Value = resp.json().await.context("Failed to parse response")?;
            Ok(result)
        }).await
    }

    /// Chat completion (streaming) — returns a byte stream.
    pub async fn chat_completion_stream(
        &self,
        body: Value,
    ) -> Result<reqwest::Response> {
        self.request_with_circuit("chat_completion_stream", async {
            let url = format!("{}/v1/chat/completions", self.base_url);
            let mut req = self
                .client
                .post(&url)
                .json(&body)
                .timeout(Duration::from_secs(600));
            if let Some(key) = &self.api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req.send().await.context("Streaming chat completion request failed")?;
            Ok(resp)
        }).await
    }

    /// Text completion (non-streaming).
    pub async fn completion(&self, body: Value) -> Result<Value> {
        let url = format!("{}/v1/completions", self.base_url);
        let mut req = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(600));
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        let resp = req.send().await.context("Completion request failed")?;
        let result: Value = resp.json().await.context("Failed to parse completion response")?;
        Ok(result)
    }
}
