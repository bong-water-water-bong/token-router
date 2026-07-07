//! 1bit NPU Server — fast Rust server wrapping npu_engine_mt.
//!
//! Replaces the Python npu_server.py with:
//! - axum async HTTP (concurrent request handling)
//! - Native Qwen3 tokenizer (no subprocess for encoding)
//! - Process pool for engine subprocesses
//! - Proper error handling and metrics

use axum::{
    Router, extract::State, http::StatusCode, response::{sse::{Event, Sse}, IntoResponse, Response},
    routing::{get, post}, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

// ─── Config ──────────────────────────────────────────────────────────

const ENGINE: &str = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";
const MODEL_PATH: &str = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
const TOKENIZER_JSON: &str = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json";
const PORT: u16 = 8081;
const MAX_CONCURRENT: usize = 4;
const MAX_TOKENS: usize = 64;

// ─── Tokenizer ──────────────────────────────────────────────────────

struct Tokenizer {
    vocab: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Tokenizer {
    fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
        let parsed: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| format!("parse: {}", e))?;

        let vocab_obj = parsed["model"]["vocab"].as_object()
            .ok_or("no vocab")?;
        let mut vocab = std::collections::HashMap::new();
        let mut id_to_token: Vec<(u32, String)> = Vec::new();

        for (tok, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as u32;
                vocab.insert(tok.clone(), id);
                id_to_token.push((id, tok.clone()));
            }
        }

        // Added tokens
        if let Some(added) = parsed["added_tokens"].as_array() {
            for t in added {
                if let (Some(id), Some(content)) =
                    (t["id"].as_u64(), t["content"].as_str())
                {
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

        info!("Tokenizer: {} vocab entries, {} tokens", vocab.len(), tokens.len());
        Ok(Self { vocab, id_to_token: tokens })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Simple BPE-style encoding for Qwen3
        let mut ids = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut found = false;
            // Try longest match first (up to 32 chars)
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
                // Fall back to single char or skip
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

// ─── Engine ─────────────────────────────────────────────────────────

struct NpuEngine {
    sem: Arc<Semaphore>,
}

impl NpuEngine {
    fn new() -> Self {
        Self { sem: Arc::new(Semaphore::new(MAX_CONCURRENT)) }
    }

    async fn infer(&self, input_tokens: &[u32], gen_count: usize) -> Result<Vec<u32>, String> {
        let _permit = self.sem.acquire().await.map_err(|_| "overloaded".to_string())?;

        let tok_str: String = input_tokens.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let start = Instant::now();
        let output = Command::new("sudo")
            .args(["sh", "-c", &format!(
                "NPU_GEN={} {} {} {}",
                gen_count, ENGINE, MODEL_PATH, tok_str
            )])
            .output()
            .await
            .map_err(|e| format!("spawn: {}", e))?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let ids: Vec<u32> = stdout
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .map(|i| i as u32)
            .collect();

        info!("infer: {}→{} tokens in {:.0}ms", input_tokens.len(), ids.len(), elapsed.as_secs_f64() * 1000.0);
        Ok(ids)
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
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: String,
    content: String,
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
    engine: NpuEngine,
}

// ─── Handlers ──────────────────────────────────────────────────────

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelEntry {
            id: "qwen3-0.6b-FLM".into(),
            object: "model".into(),
            created: std::time::UNIX_EPOCH.elapsed().unwrap().as_secs(),
            owned_by: "1bit".into(),
        }],
    })
}

async fn health() -> &'static str { "ok" }

async fn chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(16).min(MAX_TOKENS);
    let stream = req.stream.unwrap_or(false);

    // Build prompt
    let prompt = build_qwen3_prompt(&req.messages);
    let input_tokens = state.tokenizer.encode(&prompt);
    if input_tokens.is_empty() {
        return (StatusCode::BAD_REQUEST, "tokenizer failed").into_response();
    }

    // Truncate
    let input_tokens = if input_tokens.len() > 256 {
        input_tokens[..256].to_vec()
    } else {
        input_tokens
    };

    let prompt_len = input_tokens.len();

    // Run engine
    let all_ids = match state.engine.infer(&input_tokens, max_tokens).await {
        Ok(ids) => ids,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };

    let generated: Vec<u32> = if all_ids.len() > prompt_len {
        all_ids[prompt_len..].to_vec()
    } else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "no tokens").into_response();
    };

    let content = state.tokenizer.decode(&generated);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = std::time::UNIX_EPOCH.elapsed().unwrap().as_secs();

    if stream {
        let stream = tokio_stream::iter(generated.into_iter().enumerate().map(move |(i, tid)| {
            let text = state.tokenizer.decode(&[tid]);
            let finish = if i == 0 { None } else { Some("stop") };
            let chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "qwen3-0.6b-FLM",
                "choices": [{
                    "index": 0,
                    "delta": if i == 0 {
                        serde_json::json!({"role": "assistant"})
                    } else {
                        serde_json::json!({"content": text})
                    },
                    "finish_reason": finish,
                }],
            });
            Ok::<_, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
        }));
        Sse::new(stream).into_response()
    } else {
        Json(ChatResponse {
            id: completion_id,
            object: "chat.completion".into(),
            created,
            model: "qwen3-0.6b-FLM".into(),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage { role: "assistant".into(), content },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: generated.len(),
                total_tokens: prompt_len + generated.len(),
            },
        }).into_response()
    }
}

// ─── Main ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("1bit NPU Server starting...");

    let tokenizer = Tokenizer::load(TOKENIZER_JSON)
        .expect("Failed to load tokenizer");
    let engine = NpuEngine::new();

    // Pre-warm: touch the model file to keep it in page cache
    info!("Pre-warming model...");
    let _ = engine.infer(&[1, 2, 3, 4, 5], 1).await;

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
