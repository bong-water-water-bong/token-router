//! 1bit NPU Server — wraps npu_engine_server persistent daemon.
//!
//! Starts npu_engine_server as a subprocess and communicates via its
//! stdin/stdout JSON protocol. Provides an OpenAI-compatible HTTP API
//! with per-token logprobs returned in the response.
//!
//! NPU engine protocol:
//!   Input:  {"tokens":[t0,t1,...],"max_new_tokens":N}
//!   Output: {"tokens":[g0,g1,...],"logprobs":[lp0,lp1,...]}
//!
//! The logprobs are critical for:
//!   - Cascade strategy: confidence-based routing (NPU→GPU on low conf)
//!   - Speculative decode: rejection sampling (draft vs verify)

use axum::{
    Router, extract::State, http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post}, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Semaphore};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

// ─── Config ──────────────────────────────────────────────────────────

const ENGINE: &str = "/home/bcloud/engine/npu/build/npu_engine_server";
const PORT: u16 = 8081;
const MAX_CONCURRENT: usize = 4;

// ─── Engine Subprocess ───────────────────────────────────────────────

/// A persistent connection to npu_engine_server.
///
/// The C++ daemon loads the model once (~8s) and stays warm.
/// Each request is one JSON line on stdin, one JSON line on stdout.
struct EngineProc {
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
    _child: Child,
}

impl EngineProc {
    /// Spawn npu_engine_server and wait for it to signal ready.
    async fn spawn() -> Result<Self, String> {
        let mut child = Command::new("sudo")
            .args([ENGINE])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn engine: {}", e))?;

        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let reader = BufReader::new(stdout);

        // Give the engine a moment to initialize
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        Ok(Self { stdin, reader, _child: child })
    }

    /// Send a JSON request line and parse the JSON response.
    async fn send_request(&mut self, request: &str) -> Result<EngineResponse, String> {
        let start = Instant::now();

        // Write request line
        self.stdin
            .write_all(request.as_bytes())
            .await
            .map_err(|e| format!("write: {}", e))?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| format!("write newline: {}", e))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| format!("flush: {}", e))?;

        // Read response line
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .await
            .map_err(|e| format!("read: {}", e))?;

        let elapsed = start.elapsed();
        let line = line.trim();

        // Parse JSON
        let parsed: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("parse JSON: {} — raw: {}", e, line))?;

        // Check for error
        if let Some(err) = parsed.get("error").and_then(|e| e.as_str()) {
            return Err(format!("engine error: {}", err));
        }

        // Extract tokens and logprobs
        let tokens: Vec<u32> = parsed["tokens"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|u| u as u32)).collect())
            .unwrap_or_default();

        let logprobs: Vec<f64> = parsed["logprobs"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        info!(
            "engine: {}→{} tokens in {:.0}ms",
            tokens.len() - logprobs.len(), // prompt tokens (input length unknown here)
            tokens.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(EngineResponse { tokens, logprobs })
    }
}

/// Response from the NPU engine.
#[derive(Debug)]
struct EngineResponse {
    tokens: Vec<u32>,
    logprobs: Vec<f64>,
}

// ─── Engine Pool ─────────────────────────────────────────────────────

/// A pool of NPU engine subprocesses for concurrent requests.
struct EnginePool {
    procs: Mutex<Vec<EngineProc>>,
    sem: Arc<Semaphore>,
}

impl EnginePool {
    async fn new(size: usize) -> Result<Self, String> {
        let mut procs = Vec::with_capacity(size);
        for i in 0..size {
            info!("Spawning engine process {}/{}...", i + 1, size);
            let proc = EngineProc::spawn().await.map_err(|e| {
                format!("engine proc {} failed: {}", i + 1, e)
            })?;
            procs.push(proc);
        }
        info!("Engine pool ready: {} processes", procs.len());
        Ok(Self {
            procs: Mutex::new(procs),
            sem: Arc::new(Semaphore::new(size)),
        })
    }

    /// Acquire an engine process, send a request, and return the response.
    async fn infer(&self, input_tokens: &[u32], gen_count: usize) -> Result<EngineResponse, String> {
        let _permit = self
            .sem
            .acquire()
            .await
            .map_err(|_| "overloaded".to_string())?;

        let request = serde_json::json!({
            "tokens": input_tokens,
            "max_new_tokens": gen_count,
        })
        .to_string();

        let mut procs = self.procs.lock().await;
        // Round-robin: try each proc until one succeeds
        for idx in 0..procs.len() {
            let proc = &mut procs[idx];
            match proc.send_request(&request).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    warn!("engine proc {} failed: {}, trying next", idx, e);
                }
            }
        }
        Err("all engine processes failed".to_string())
    }
}

// ─── Tokenizer ──────────────────────────────────────────────────────

struct Tokenizer {
    vocab: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Tokenizer {
    fn load(path: &str) -> Result<Self, String> {
        let data =
            std::fs::read_to_string(path).map_err(|e| format!("read tokenizer: {}", e))?;
        let parsed: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| format!("parse tokenizer: {}", e))?;

        let vocab_obj = parsed["model"]["vocab"]
            .as_object()
            .ok_or("no vocab in tokenizer")?;
        let mut vocab = std::collections::HashMap::new();
        let mut id_to_token: Vec<(u32, String)> = Vec::new();

        for (tok, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as u32;
                vocab.insert(tok.clone(), id);
                id_to_token.push((id, tok.clone()));
            }
        }

        if let Some(added) = parsed["added_tokens"].as_array() {
            for t in added {
                if let (Some(id), Some(content)) = (t["id"].as_u64(), t["content"].as_str()) {
                    vocab.insert(content.to_string(), id as u32);
                    id_to_token.push((id as u32, content.to_string()));
                }
            }
        }

        id_to_token.sort_by_key(|(id, _)| *id);
        let max_id = id_to_token.last().map(|(id, _)| *id).unwrap_or(0) as usize;
        let mut tokens = vec!["<unk>".to_string(); max_id + 1];
        for (id, tok) in &id_to_token {
            if (*id as usize) < tokens.len() {
                tokens[*id as usize] = tok.clone();
            }
        }

        info!(
            "Tokenizer: {} vocab entries, {} tokens mapped",
            vocab.len(),
            tokens.len()
        );
        Ok(Self { vocab, id_to_token: tokens })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut found = false;
            for len in (1..=32.min(chars.len() - i)).rev() {
                let candidate: String = chars[i..i + len].iter().collect();
                if self.vocab.contains_key(&candidate) {
                    ids.push(self.vocab[&candidate]);
                    i += len;
                    found = true;
                    break;
                }
            }
            if !found {
                let c = chars[i].to_string();
                if let Some(&id) = self.vocab.get(&c) {
                    ids.push(id);
                }
                i += 1;
            }
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .cloned()
            .collect()
    }
}

// ─── Chat Template ──────────────────────────────────────────────────

fn build_qwen3_prompt(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => parts.push(format!("<|im_start|>system\n{}<|im_end|>", msg.content)),
            "user" => parts.push(format!("<|im_start|>user\n{}<|im_end|>", msg.content)),
            "assistant" => parts.push(format!("<|im_start|>assistant\n{}<|im_end|>", msg.content)),
            _ => {}
        }
    }
    parts.push("<|im_start|>assistant\n".to_string());
    parts.join("\n")
}

// ─── API Types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    message: AssistantMessage,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ResponseLogprobs>,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseLogprobs {
    content: Vec<TokenLogprob>,
}

#[derive(Debug, Serialize)]
struct TokenLogprob {
    token: String,
    logprob: f64,
    bytes: Option<Vec<u64>>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelEntry>,
}

// ─── App State ─────────────────────────────────────────────────────

struct AppState {
    tokenizer: Tokenizer,
    engine: EnginePool,
}

// ─── Handlers ──────────────────────────────────────────────────────

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelEntry {
            id: "qwen3-0.6b-FLM".into(),
            object: "model".into(),
            created: std::time::UNIX_EPOCH
                .elapsed()
                .unwrap()
                .as_secs(),
            owned_by: "1bit".into(),
        }],
    })
}

async fn health() -> &'static str {
    "ok"
}

async fn chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(16).min(256);
    let stream = req.stream.unwrap_or(false);
    let want_logprobs = req.logprobs.unwrap_or(false);

    // Build prompt
    let prompt = build_qwen3_prompt(&req.messages);
    let input_tokens = state.tokenizer.encode(&prompt);
    if input_tokens.is_empty() {
        return (StatusCode::BAD_REQUEST, "tokenizer failed").into_response();
    }

    // Truncate to engine's max context
    let input_tokens = if input_tokens.len() > 256 {
        input_tokens[..256].to_vec()
    } else {
        input_tokens
    };

    let prompt_len = input_tokens.len();

    // Run engine
    let engine_resp = match state.engine.infer(&input_tokens, max_tokens).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };

    // Separate prompt tokens from generated tokens
    // The engine returns ALL output tokens (prompt + generated after decode)
    let generated_ids: Vec<u32> = if engine_resp.tokens.len() > prompt_len {
        engine_resp.tokens[prompt_len..].to_vec()
    } else {
        engine_resp.tokens.clone()
    };

    // Logprobs only cover generated tokens
    let generated_logprobs: Vec<f64> = engine_resp.logprobs;

    // Build token-level logprob info for OpenAI response
    let logprob_content: Vec<TokenLogprob> = if want_logprobs {
        generated_ids
            .iter()
            .zip(generated_logprobs.iter().chain(std::iter::repeat(&0.0)))
            .map(|(tid, lp)| {
                let token = state.tokenizer.decode(&[*tid]);
                TokenLogprob {
                    token,
                    logprob: *lp,
                    bytes: None,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let content = state.tokenizer.decode(&generated_ids);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = std::time::UNIX_EPOCH.elapsed().unwrap().as_secs();

    // ── Streaming response (SSE) ───────────────────────────────────
    if stream {
        use axum::response::sse::{Event, Sse};
        use std::convert::Infallible;

        let num_tokens = generated_ids.len();
        let stream = tokio_stream::iter(
            generated_ids
                .into_iter()
                .enumerate()
                .map(move |(i, tid)| {
                    let text = state.tokenizer.decode(&[tid]);
                    let is_last = i == num_tokens - 1;
                    let finish = if is_last { Some("stop") } else { None };

                    // Build per-token logprobs for SSE
                    let token_lp = if want_logprobs {
                        let lp = generated_logprobs.get(i).copied().unwrap_or(0.0);
                        let tok = state.tokenizer.decode(&[tid]);
                        Some(serde_json::json!({
                            "content": [{
                                "token": tok,
                                "logprob": lp,
                                "bytes": null,
                            }]
                        }))
                    } else {
                        None
                    };

                    let chunk = serde_json::json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "qwen3-0.6b-FLM",
                        "choices": [{
                            "index": 0,
                            "delta": if i == 0 {
                                serde_json::json!({"role": "assistant", "content": text})
                            } else if is_last && text.is_empty() {
                                serde_json::json!({})
                            } else {
                                serde_json::json!({"content": text})
                            },
                            "finish_reason": finish,
                            "logprobs": token_lp,
                        }],
                    });

                    Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap())
                }),
        );
        Sse::new(stream).into_response()
    }
    // ── Non-streaming response ─────────────────────────────────────
    else {
        Json(ChatResponse {
            id: completion_id,
            object: "chat.completion".into(),
            created,
            model: "qwen3-0.6b-FLM".into(),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: "stop".into(),
                logprobs: if want_logprobs {
                    Some(ResponseLogprobs {
                        content: logprob_content,
                    })
                } else {
                    None
                },
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: generated_ids.len(),
                total_tokens: prompt_len + generated_ids.len(),
            },
        })
        .into_response()
    }
}

// ─── Main ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("1bit NPU Server starting...");

    let tokenizer = Tokenizer::load(
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json",
    )
    .expect("Failed to load tokenizer");

    info!("Starting NPU engine pool ({} processes)...", MAX_CONCURRENT);
    let engine = EnginePool::new(MAX_CONCURRENT)
        .await
        .expect("Failed to start engine pool");

    let state = Arc::new(AppState { tokenizer, engine });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST]);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completion))
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{}", PORT);
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
