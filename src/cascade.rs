//! Cascade streaming engine — token-level confidence-based routing.
//!
//! The core innovation: stream from the fast backend (NPU), parse each SSE
//! chunk for token content and log-probability. When confidence drops below
//! threshold, seamlessly switch to the capable backend (GPU) — all within
//! a single client-facing SSE stream.
//!
//! ```text
//! NPU stream → parse tokens + logprobs → confidence OK? → forward to client
//!                                       → low confidence → switch to GPU → forward
//! ```

use crate::backend::BackendClient;
use crate::stream::{extract_delta, extract_log_prob, parse_sse_chunk};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Run a cascade streaming session.
///
/// Returns an infallible byte stream (`Result<Bytes, Infallible>`) suitable
/// for `axum::body::Body::from_stream()`.
pub fn cascade_stream(
    npu_client: Arc<BackendClient>,
    gpu_client: Arc<BackendClient>,
    body: Value,
    threshold: f64,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> {
    let (tx, rx) = mpsc::channel::<Bytes>(128);

    tokio::spawn(async move {
        if let Err(e) = run_cascade(npu_client, gpu_client, body, threshold, tx).await {
            warn!("Cascade stream error: {}", e);
        }
    });

    tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok)
}

/// Internal cascade loop.
async fn run_cascade(
    npu_client: Arc<BackendClient>,
    gpu_client: Arc<BackendClient>,
    mut body: Value,
    threshold: f64,
    tx: mpsc::Sender<Bytes>,
) -> Result<(), String> {
    // Inject logprobs requirement
    body["logprobs"] = Value::Bool(true);
    body["top_logprobs"] = Value::Number(serde_json::Number::from(1));
    body["stream"] = Value::Bool(true);

    // ── Phase 1: Stream from NPU ──────────────────────────────────────
    info!("Cascade: starting NPU stream");
    let npu_resp = npu_client
        .chat_completion_stream(body.clone())
        .await
        .map_err(|e| format!("NPU request failed: {}", e))?;

    let mut npu_stream = npu_resp.bytes_stream();
    let mut sse_buf = String::new();
    let mut accumulated: Vec<String> = Vec::new();
    let mut recent_logprobs: Vec<f64> = Vec::new();  // sliding window for confidence
    const CONFIDENCE_WINDOW: usize = 3;

    while let Some(chunk_result) = npu_stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("NPU stream error: {}", e))?;
        let chunk_str = String::from_utf8_lossy(&chunk).to_string();
        sse_buf.push_str(&chunk_str);

        // Process complete SSE events (separated by \n\n)
        while let Some(pos) = sse_buf.find("\n\n") {
            let event = sse_buf[..pos].to_string();
            sse_buf = sse_buf[pos + 2..].to_string();

            for line in event.lines() {
                let trimmed = line.trim();
                let Some(data_str) = parse_sse_chunk(trimmed) else { continue };
                if data_str == "[DONE]" {
                    let _ = tx.send(Bytes::from("data: [DONE]\n\n")).await;
                    return Ok(());
                }

                let delta = extract_delta(&data_str);
                let logprob = extract_log_prob(&data_str);

                // Track accumulated tokens and sliding window logprobs
                if let Some(tok) = &delta {
                    accumulated.push(tok.clone());
                }
                if let Some(lp) = logprob {
                    recent_logprobs.push(lp as f64);
                    if recent_logprobs.len() > CONFIDENCE_WINDOW {
                        recent_logprobs.remove(0);
                    }
                }

                // Check confidence using sliding window average (reduces noise)
                // Only trigger after we have enough tokens and logprobs
                if recent_logprobs.len() >= CONFIDENCE_WINDOW && accumulated.len() >= CONFIDENCE_WINDOW {
                    let avg_logprob: f64 = recent_logprobs.iter().sum::<f64>() / recent_logprobs.len() as f64;
                    if avg_logprob < threshold {
                        info!(
                            "Cascade: switch to GPU at token {} (avg_logprob={:.3} < {:.1}, window={})",
                            accumulated.len(),
                            avg_logprob,
                            threshold,
                            CONFIDENCE_WINDOW
                        );

                        // Handoff to GPU with accumulated context
                        let gpu_resp = start_gpu(gpu_client.clone(), &body, &accumulated)
                            .await
                            .map_err(|e| format!("GPU handoff failed: {}", e))?;

                        // Forward GPU stream to client, then exit
                        forward_gpu(gpu_resp, &tx, &accumulated).await;
                        return Ok(());
                    }
                }

                // Forward the raw chunk
                let _ = tx.send(chunk.clone()).await;
            }
        }
    }

    // NPU stream ended naturally — send [DONE]
    let _ = tx.send(Bytes::from("data: [DONE]\n\n")).await;
    Ok(())
}

/// Start a GPU request with accumulated NPU context appended as assistant message.
async fn start_gpu(
    gpu_client: Arc<BackendClient>,
    original: &Value,
    accumulated: &[String],
) -> Result<reqwest::Response, String> {
    let mut gpu_body = original.clone();
    let assistant_text = accumulated.concat();

    if !assistant_text.is_empty() {
        if let Some(messages) = gpu_body["messages"].as_array_mut() {
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": assistant_text
            }));
        }
    }

    gpu_body["stream"] = Value::Bool(true);
    gpu_body["logprobs"] = Value::Bool(false);

    info!("Cascade: starting GPU stream");
    gpu_client
        .chat_completion_stream(gpu_body)
        .await
        .map_err(|e| format!("GPU request failed: {}", e))
}

/// Forward GPU's SSE stream chunks into the output channel.
async fn forward_gpu(
    gpu_resp: reqwest::Response,
    tx: &mpsc::Sender<Bytes>,
    _accumulated: &[String],
) {
    let mut gpu_stream = gpu_resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = gpu_stream.next().await {
        match chunk {
            Ok(bytes) => {
                let s = String::from_utf8_lossy(&bytes).to_string();
                buf.push_str(&s);

                // Emit complete SSE events
                while let Some(pos) = buf.find("\n\n") {
                    let event = buf[..pos + 2].to_string();
                    buf = buf[pos + 2..].to_string();

                    if event.contains("[DONE]") {
                        let _ = tx.send(Bytes::from("data: [DONE]\n\n")).await;
                        return;
                    }
                    let _ = tx.send(Bytes::from(event)).await;
                }
            }
            Err(e) => {
                warn!("GPU stream error: {}", e);
                break;
            }
        }
    }

    let _ = tx.send(Bytes::from("data: [DONE]\n\n")).await;
}
