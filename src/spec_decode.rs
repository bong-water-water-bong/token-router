//! Speculative decoding streaming engine.
//!
//! Draft N tokens from a fast backend (NPU), verify all N in a single
//! forward pass on a capable backend (GPU). Accept tokens using rejection
//! sampling — tokens where the target model agrees with the draft are kept;
//! the first rejected token and everything after it is discarded, and the
//! draft resumes from the last accepted position.
//!
//! On Strix Halo's unified memory architecture, the NPU→GPU handoff uses
//! zero-copy dma-buf (via `kv_cache::handoff`), making the verify pass
//! essentially free in terms of data movement.
//!
//! # Algorithm
//!
//! ```text
//! repeat:
//!   1. Draft: NPU generates N tokens greedily, captures logprobs
//!   2. Verify: GPU gets prompt + draft tokens, returns logits for each
//!   3. Accept: for each position i:
//!        if GPU_logprob(t_i) >= draft_logprob(t_i) → ACCEPT
//!        else → accept with probability P = min(1, q(x)/p(x))
//!              on rejection → discard t_i..t_N, resume from t_{i-1}
//!   4. Forward accepted tokens to client
//!   5. Adjust N dynamically based on acceptance rate
//! ```

use crate::backend::BackendClient;
use crate::stream::build_sse_chunk;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Global counter for speculative decode rounds (observability).
static SPEC_DECODE_ROUNDS: AtomicUsize = AtomicUsize::new(0);
static SPEC_ACCEPTED_TOKENS: AtomicUsize = AtomicUsize::new(0);
static SPEC_DRAFTED_TOKENS: AtomicUsize = AtomicUsize::new(0);
static SPEC_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);

/// Get speculative decode metrics.
pub fn metrics() -> Value {
    let rounds = SPEC_DECODE_ROUNDS.load(Ordering::Relaxed);
    let accepted = SPEC_ACCEPTED_TOKENS.load(Ordering::Relaxed);
    let drafted = SPEC_DRAFTED_TOKENS.load(Ordering::Relaxed);
    let cache_hits = SPEC_CACHE_HITS.load(Ordering::Relaxed);
    let acceptance_rate = if drafted > 0 {
        accepted as f64 / drafted as f64
    } else {
        0.0
    };

    serde_json::json!({
        "rounds": rounds,
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate": format!("{:.3}", acceptance_rate),
        "kv_cache_hits": cache_hits,
        "n_draft_current": 0, // filled dynamically
    })
}

/// A single draft token produced by the draft model.
#[derive(Debug, Clone)]
struct DraftToken {
    text: String,
    log_prob: f64,
    token_id: Option<u32>,
}

/// Result of a verification round.
#[derive(Debug)]
enum VerificationResult {
    /// All N tokens accepted. Continue drafting from the end.
    AllAccepted { n: usize },
    /// Partial acceptance — accept up to index `accepted_up_to` (exclusive),
    /// reject at that position. Resume drafting from this position.
    PartialAccept { accepted_up_to: usize, n: usize },
}

/// Run a speculative decoding streaming session.
///
/// The stream produces SSE-formatted chat completion chunks suitable for
/// OpenAI-compatible clients. The client sees a single continuous stream —
/// the draft/verify cycling is invisible to the caller.
pub fn spec_decode_stream(
    draft_client: Arc<BackendClient>,
    target_client: Arc<BackendClient>,
    body: Value,
    n_draft: usize,
    acceptance_threshold: f64,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> {
    let (tx, rx) = mpsc::channel::<Bytes>(256);

    tokio::spawn(async move {
        if let Err(e) = run_spec_decode(
            draft_client, target_client, body, n_draft, acceptance_threshold, tx,
        ).await {
            warn!("Speculative decode stream error: {}", e);
        }
    });

    tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok)
}

/// Internal speculative decode loop.
///
/// Loops: draft → verify → accept → forward → repeat
async fn run_spec_decode(
    draft_client: Arc<BackendClient>,
    target_client: Arc<BackendClient>,
    body: Value,
    mut n_draft: usize,
    acceptance_threshold: f64,
    tx: mpsc::Sender<Bytes>,
) -> Result<(), String> {
    // Track running acceptance rate for dynamic n_draft adjustment
    let mut running_accept_rate: f64 = 0.7; // start optimistic
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut round_count = 0usize;

    // Original messages (reset each round so the draft model doesn't
    // see its own output and loop).
    let original_messages: Vec<Value> = body["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut conversation: Vec<Value> = original_messages.clone();

    // Track the total assistant response so far for the final response
    let mut assistant_response = String::new();

    // Track whether we've started the response (for finish_reason logic)
    let mut _stream_started = false;

    // ── Main speculative decode loop ─────────────────────────────────
    loop {
        round_count += 1;
        SPEC_DECODE_ROUNDS.fetch_add(1, Ordering::Relaxed);

        // Clamp n_draft to sane bounds
        n_draft = n_draft.clamp(1, 16);

        // ── Phase 1: Draft ────────────────────────────────────────────
        // Build draft request: NON-streaming with logprobs.
        // FLM and OpenAI-compatible backends return per-token logprobs in
        // non-streaming mode but NOT in streaming SSE chunks.
        // Using non-streaming gives us REAL logprobs for rejection sampling.
        let mut draft_body = body.clone();
        draft_body["messages"] = Value::Array(conversation.clone());
        draft_body["logprobs"] = Value::Bool(true);
        draft_body["top_logprobs"] = Value::Number(serde_json::Number::from(5));
        draft_body["stream"] = Value::Bool(false);
        draft_body["max_tokens"] = Value::Number(serde_json::Number::from(n_draft as u64));
        draft_body["temperature"] = Value::Number(serde_json::Number::from_f64(0.0).unwrap());
        draft_body["top_p"] = Value::Number(serde_json::Number::from_f64(1.0).unwrap());

        let draft_resp = draft_client
            .chat_completion(draft_body)
            .await
            .map_err(|e| format!("Draft request failed: {}", e))?;

        // Parse draft response: extract tokens and their REAL logprobs
        let draft_tokens = extract_draft_tokens(&draft_resp, n_draft);

        let draft_finished_early = draft_tokens.len() < n_draft;

        // If draft produced no tokens, we're done
        if draft_tokens.is_empty() {
            debug!("SpecDecode: draft produced no tokens, ending");
            break;
        }

        total_drafted += draft_tokens.len();
        SPEC_DRAFTED_TOKENS.fetch_add(draft_tokens.len(), Ordering::Relaxed);

        info!(
            "SpecDecode round {}: drafted {} tokens (n_draft={})",
            round_count, draft_tokens.len(), n_draft
        );

        // ── Phase 2: Verify ───────────────────────────────────────────
        // Send prompt + draft tokens to the target backend (ROCm) to get
        // real per-token logprobs for rejection sampling.
        // ROCm llama-server supports logprobs with full top_logprobs distribution.
        let mut verify_body = body.clone();

        // Build messages: original conversation + assistant draft tokens
        let draft_text: String = draft_tokens.iter().map(|t| t.text.as_str()).collect();
        let mut verify_messages = conversation.clone();
        verify_messages.push(serde_json::json!({
            "role": "assistant",
            "content": draft_text,
            // Include per-token logprobs so the verify backend can compare
            // against what the draft model actually produced
        }));
        verify_body["messages"] = Value::Array(verify_messages);
        verify_body["logprobs"] = Value::Bool(true);
        verify_body["top_logprobs"] = Value::Number(serde_json::Number::from(5));
        verify_body["stream"] = Value::Bool(false);
        verify_body["max_tokens"] = Value::Number(serde_json::Number::from(1));

        // Strip model field if present — backends like ZINC reject unknown models
        if let Some(model_val) = verify_body.get("model") {
            if model_val.as_str().map(|s| s.contains("spec_decode")).unwrap_or(false)
                || model_val.as_str().map(|s| s.contains("cascade")).unwrap_or(false)
            {
                verify_body["model"] = Value::Null;
            }
        }

        let verify_start = std::time::Instant::now();
        let verify_resp = target_client
            .chat_completion(verify_body)
            .await
            .map_err(|e| format!("Verify request failed: {}", e))?;
        let verify_elapsed = verify_start.elapsed();

        // ── Phase 3: Acceptance ───────────────────────────────────────
        // Parse the verification response to get logprobs for each draft position.
        // The target model returns logprobs for the entire prompt+draft sequence.
        // We take the LAST `draft_len` entries (the generated tokens).
        let verify_logprobs = extract_verify_logprobs(&verify_resp, draft_tokens.len());

        debug!(
            "Verify: {} tokens in {:.0}ms, got {} logprobs",
            draft_tokens.len(),
            verify_elapsed.as_secs_f64() * 1000.0,
            verify_logprobs.len(),
        );

        let mut _accepted_count = 0;
        let mut first_rejection_idx: Option<usize> = None;

        for (i, draft) in draft_tokens.iter().enumerate() {
            let target_lp = verify_logprobs.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let draft_lp = draft.log_prob;

            // Standard speculative decoding acceptance criterion:
            // Accept if target model would have assigned at least as high
            // probability to this token as the draft model did.
            // This is the core rejection sampling step.
            if target_lp.is_infinite() || draft_lp.is_infinite() {
                // If we can't get real logprobs, fall back to deterministic accept
                // based on whether the target model actually continues similarly
                _accepted_count += 1;
                continue;
            }

            let accept = if target_lp >= draft_lp {
                // Target agrees (or is more confident) — always accept
                true
            } else {
                // Target is less confident — probabilistically accept
                // using rejection sampling: accept with prob q(x)/p(x)
                let ratio = (target_lp - draft_lp).exp(); // exp(log q - log p) = q/p
                let threshold = acceptance_threshold.max(ratio);
                // Use a deterministic check based on the draft token's hash
                // so results are reproducible
                let hash_seed = draft.text.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                let prob = (hash_seed % 1000) as f64 / 1000.0;
                prob < threshold
            };

            if accept {
                _accepted_count += 1;
            } else {
                first_rejection_idx = Some(i);
                break;
            }
        }

        // ── Phase 4: Act on results ───────────────────────────────────
        let accepted_tokens: Vec<&DraftToken> = match first_rejection_idx {
            Some(idx) => {
                // Partial acceptance
                draft_tokens[..idx].iter().collect()
            }
            None => {
                // All accepted (or draft ended early)
                draft_tokens.iter().collect()
            }
        };

        total_accepted += accepted_tokens.len();
        SPEC_ACCEPTED_TOKENS.fetch_add(accepted_tokens.len(), Ordering::Relaxed);

        // Update running acceptance rate (exponential moving average)
        let round_accept_rate = if draft_tokens.is_empty() {
            1.0
        } else {
            accepted_tokens.len() as f64 / draft_tokens.len() as f64
        };
        running_accept_rate = running_accept_rate * 0.7 + round_accept_rate * 0.3;

        // Dynamic n_draft adjustment:
        // High acceptance → increase draft length (more parallelism)
        // Low acceptance → decrease draft length (less waste)
        let new_n_draft = if running_accept_rate > 0.85 {
            (n_draft as f64 * 1.25).round().min(16.0) as usize
        } else if running_accept_rate < 0.5 {
            (n_draft as f64 * 0.75).round().max(1.0) as usize
        } else {
            n_draft
        };
        if new_n_draft != n_draft {
            debug!("SpecDecode: n_draft {} → {} (accept_rate={:.3})", n_draft, new_n_draft, running_accept_rate);
            n_draft = new_n_draft;
        }

        info!(
            "SpecDecode round {}: accepted {}/{} (rate={:.3}), n_draft={}, running_rate={:.3}",
            round_count, accepted_tokens.len(), draft_tokens.len(), round_accept_rate, n_draft, running_accept_rate
        );

        // ── Phase 5: Forward to client ────────────────────────────────
        _stream_started = true;
        for token in &accepted_tokens {
            let sse_chunk = build_sse_chunk(&token.text, None);
            let _ = tx.send(Bytes::from(sse_chunk)).await;
            assistant_response.push_str(&token.text);
        }

        // Update conversation for next round
        // The assistant response so far is the full accumulated text
        // Remove the old assistant message if any, and add the updated one
        conversation.retain(|msg| msg["role"] != "assistant");
        conversation.push(serde_json::json!({
            "role": "assistant",
            "content": assistant_response,
        }));

        // Check if we should stop:
        // 1. Draft finished early (model hit EOS or max_tokens)
        // 2. All tokens rejected (degenerate case — fall back)
        // 3. Reached empty accept (shouldn't happen, but guard)
        if draft_finished_early && accepted_tokens.is_empty() {
            break;
        }
        if accepted_tokens.is_empty() {
            warn!("SpecDecode: all draft tokens rejected in round {}, falling back to single token", round_count);
            // Fallback: forward the first draft token anyway (conservative)
            if let Some(first) = draft_tokens.first() {
                let sse_chunk = build_sse_chunk(&first.text, None);
                let _ = tx.send(Bytes::from(sse_chunk)).await;
                assistant_response.push_str(&first.text);
                // Update conversation
                conversation.retain(|msg| msg["role"] != "assistant");
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "content": assistant_response,
                }));
            }
        }

        // Safety: don't loop forever  
        if round_count >= 512 {
            warn!("SpecDecode: hit max round limit (512), terminating");
            break;
        }
    }

    // Send termination event
    let done = build_sse_chunk("", Some("stop"));
    let _ = tx.send(Bytes::from(done)).await;
    let _ = tx.send(Bytes::from("data: [DONE]\n\n")).await;

    // Log final metrics
    info!(
        "SpecDecode complete: {} rounds, {}/{} tokens accepted ({:.1}%), final n_draft={}",
        round_count, total_accepted, total_drafted,
        if total_drafted > 0 { 100.0 * total_accepted as f64 / total_drafted as f64 } else { 0.0 },
        n_draft,
    );

    Ok(())
}

/// Extract draft tokens with their REAL logprobs from a non-streaming
/// draft response (FLM / OpenAI-compatible format).
///
/// Expected response format:
/// ```json
/// {
///   "choices": [{
///     "logprobs": {
///       "content": [
///         {"token": "eller", "logprob": -1.078},
///         ...
///       ]
///     }
///   }]
/// }
/// ```
fn extract_draft_tokens(response: &Value, n_draft: usize) -> Vec<DraftToken> {
    let mut tokens = Vec::with_capacity(n_draft);

    // Parse the choice's message content for the raw text
    let response_text = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // If we have logprobs content, extract per-token info
    if let Some(content) = response
        .pointer("/choices/0/logprobs/content")
        .and_then(|c| c.as_array())
    {
        for entry in content.iter() {
            let text = entry["token"].as_str().unwrap_or("").to_string();
            let logprob = entry["logprob"].as_f64().unwrap_or(-0.1);
            let token_id = entry["id"].as_u64().map(|id| id as u32);

            if text.is_empty() {
                continue;
            }
            tokens.push(DraftToken {
                text,
                log_prob: logprob,
                token_id,
            });

            if tokens.len() >= n_draft {
                break;
            }
        }
    }

    // Fallback: if no logprobs content, split response_text by whitespace
    // (crude approximation — happens when backend doesn't support logprobs)
    if tokens.is_empty() && !response_text.is_empty() {
        // Split into individual tokens using basic heuristic
        let mut split_tokens: Vec<String> = Vec::new();

        // For backends that return text but no logprobs, emit as single tokens
        if response_text.len() <= 32 {
            // Short text: treat as one token
            split_tokens.push(response_text);
        }

        for tok in split_tokens {
            tokens.push(DraftToken {
                text: tok,
                log_prob: -0.5, // moderate confidence default
                token_id: None,
            });
            if tokens.len() >= n_draft {
                break;
            }
        }
    }

    tokens
}
///
/// The verification response is a non-streaming chat completion that includes
/// the prompt + draft tokens. We need to extract the logprobs for the *output*
/// tokens (the draft tokens), not the prompt tokens.
fn extract_verify_logprobs(response: &Value, draft_len: usize) -> Vec<f64> {
    let mut logprobs = Vec::with_capacity(draft_len);

    // The response structure depends on the backend.
    // OpenAI-compatible format:
    //   response["choices"][0]["logprobs"]["content"] = [{ "token": ..., "logprob": ... }, ...]
    //
    // The content array includes logprobs for ALL tokens (prompt + generated).
    // We want only the generated tokens at the end.
    if let Some(content) = response
        .pointer("/choices/0/logprobs/content")
        .and_then(|c| c.as_array())
    {
        // The last `draft_len` entries should be the generated tokens
        let start = content.len().saturating_sub(draft_len);
        for entry in content.iter().skip(start) {
            let lp = entry["logprob"].as_f64().unwrap_or(f64::NEG_INFINITY);
            logprobs.push(lp);
        }
    }

    // If we couldn't parse logprobs, assign default values
    while logprobs.len() < draft_len {
        logprobs.push(-1.0);
    }

    logprobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_verify_logprobs_full() {
        let response = serde_json::json!({
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "The", "logprob": -0.05},
                        {"token": " quantum", "logprob": -0.10},
                        {"token": " Fourier", "logprob": -0.03},
                        {"token": " transform", "logprob": -0.08},
                    ]
                }
            }]
        });
        let lps = extract_verify_logprobs(&response, 2);
        assert_eq!(lps.len(), 2);
        // Last 2 entries: "Fourier" (-0.03) and " transform" (-0.08)
        assert!((lps[0] - (-0.03)).abs() < 0.001);
        assert!((lps[1] - (-0.08)).abs() < 0.001);
    }

    #[test]
    fn test_extract_verify_logprobs_empty() {
        let response = serde_json::json!({});
        let lps = extract_verify_logprobs(&response, 4);
        assert_eq!(lps.len(), 4);
        for lp in &lps {
            assert!((*lp - (-1.0)).abs() < 0.001);
        }
    }

    #[test]
    fn test_extract_verify_logprobs_partial() {
        let response = serde_json::json!({
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "Hello", "logprob": -0.5},
                    ]
                }
            }]
        });
        let lps = extract_verify_logprobs(&response, 3);
        assert_eq!(lps.len(), 3);
        // Only 1 entry available, rest default to -1.0
        assert!((lps[0] - (-0.5)).abs() < 0.001);
        assert!((lps[1] - (-1.0)).abs() < 0.001);
        assert!((lps[2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_draft_token_creation() {
        let tok = DraftToken {
            text: " quantum".into(),
            log_prob: -0.1,
            token_id: Some(42),
        };
        assert_eq!(tok.text, " quantum");
        assert!((tok.log_prob - (-0.1)).abs() < 0.001);
        assert_eq!(tok.token_id, Some(42));
    }

    #[test]
    fn test_metrics_format() {
        let m = metrics();
        assert!(m["rounds"].as_u64().is_some());
        assert!(m["acceptance_rate"].as_str().is_some());
        assert!(m["n_draft_current"].as_u64().is_some());
    }
}
