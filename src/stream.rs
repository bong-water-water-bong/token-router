//! Streaming response merger.
//!
//! Handles SSE (Server-Sent Events) streaming from multiple backends,
//! merging them into a single coherent stream for the client.

use bytes::Bytes;
use futures::Stream;
use pin_project::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

/// SSE event data from a backend stream.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub data: String,
    pub backend: String,
}

/// Merged SSE stream — wraps a single backend's SSE stream and optionally
/// annotates routing metadata.
#[pin_project]
pub struct ProxyStream<S> {
    #[pin]
    inner: S,
    backend_id: String,
}

impl<S> ProxyStream<S> {
    #[allow(dead_code)]
    pub fn new(inner: S, backend_id: String) -> Self {
        Self { inner, backend_id }
    }
}

impl<S> Stream for ProxyStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.inner.poll_next(cx)
    }
}

/// Parse an SSE data chunk and extract the content delta (if any).
/// Returns None for "[DONE]" or empty lines.
#[allow(dead_code)]
pub fn parse_sse_chunk(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || line == "data: [DONE]" {
        return None;
    }
    if let Some(data) = line.strip_prefix("data: ") {
        Some(data.to_string())
    } else {
        None
    }
}

/// Try to extract the token text from a chat completion chunk.
#[allow(dead_code)]
pub fn extract_delta(chunk: &str) -> Option<String> {
    use serde_json::Value;
    let parsed: Value = serde_json::from_str(chunk).ok()?;
    let delta = parsed["choices"][0]["delta"]["content"].as_str()?;
    if delta.is_empty() {
        None
    } else {
        Some(delta.to_string())
    }
}

/// Try to extract log-prob from a chunk.
#[allow(dead_code)]
pub fn extract_log_prob(chunk: &str) -> Option<f32> {
    use serde_json::Value;
    let parsed: Value = serde_json::from_str(chunk).ok()?;
    parsed["choices"][0]["logprobs"]["content"][0]["logprob"].as_f64().map(|v| v as f32)
}

/// Build an SSE data string from a chat completion delta.
#[allow(dead_code)]
pub fn build_sse_chunk(delta: &str, finish_reason: Option<&str>) -> String {
    let finish = match finish_reason {
        Some(reason) => format!("\"finish_reason\":\"{}\"", reason),
        None => "\"finish_reason\":null".to_string(),
    };
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"index\":0,\"logprobs\":null,{}}}]}}\n\n",
        serde_json::Value::String(delta.to_string()),
        finish
    )
}
