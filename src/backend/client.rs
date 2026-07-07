//! HTTP client for OpenAI-compatible inference backends.
//!
//! Wraps `reqwest` with backend-agnostic request/response handling.
//! Supports streaming (SSE) and non-streaming modes.

use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

/// Streaming chunk from an SSE response.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub data: String,
}

/// HTTP client wrapping a single backend URL.
#[derive(Debug, Clone)]
pub struct BackendClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl BackendClient {
    /// Create a new client pointing at a backend.
    pub fn new(base_url: &str, api_key: Option<&str>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("Failed to build reqwest client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(String::from),
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
    }

    /// Chat completion (streaming) — returns a byte stream.
    pub async fn chat_completion_stream(
        &self,
        body: Value,
    ) -> Result<reqwest::Response> {
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
