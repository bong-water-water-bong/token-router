//! Context manager — tracks conversation state, generated tokens, and
//! confidence signals for routing decisions.

use serde::{Deserialize, Serialize};

/// A single generated token with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub token: String,
    pub log_prob: Option<f32>,
    pub entropy: Option<f32>,
    pub backend: String,
}

/// Message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Full routing context for a single generation request.
#[derive(Debug, Clone)]
pub struct Context {
    /// Unique session ID for this request.
    #[allow(dead_code)]
    pub session_id: String,

    /// Full conversation so far (system + messages).
    pub messages: Vec<Message>,

    /// Tokens generated so far in this response.
    pub generated: Vec<GeneratedToken>,

    /// Total tokens generated in this response.
    pub total_tokens: usize,

    /// Maximum tokens allowed for this response.
    #[allow(dead_code)]
    pub max_tokens: usize,

    /// User-specified model hint (e.g., "cascade", "fast").
    #[allow(dead_code)]
    pub model_hint: Option<String>,

    /// Whether streaming is requested.
    #[allow(dead_code)]
    pub stream: bool,
}

impl Context {
    pub fn new(session_id: String, max_tokens: usize) -> Self {
        Self {
            session_id,
            messages: Vec::new(),
            generated: Vec::new(),
            total_tokens: 0,
            max_tokens,
            model_hint: None,
            stream: false,
        }
    }

    /// Add a generated token.
    #[allow(dead_code)]
    pub fn push_token(&mut self, token: String, log_prob: Option<f32>, entropy: Option<f32>, backend: String) {
        self.generated.push(GeneratedToken {
            token,
            log_prob,
            entropy,
            backend,
        });
        self.total_tokens += 1;
    }

    /// Average log-prob of the last N tokens.
    #[allow(dead_code)]
    pub fn avg_log_prob_last_n(&self, n: usize) -> Option<f32> {
        let tokens: Vec<&GeneratedToken> = self.generated.iter().rev().take(n).collect();
        let probs: Vec<f32> = tokens.iter().filter_map(|t| t.log_prob).collect();
        if probs.is_empty() {
            None
        } else {
            Some(probs.iter().sum::<f32>() / probs.len() as f32)
        }
    }

    /// Log-prob of the last generated token.
    #[allow(dead_code)]
    pub fn last_log_prob(&self) -> Option<f32> {
        self.generated.last().and_then(|t| t.log_prob)
    }

    /// Last token text.
    #[allow(dead_code)]
    pub fn last_token(&self) -> Option<&str> {
        self.generated.last().map(|t| t.token.as_str())
    }
}
