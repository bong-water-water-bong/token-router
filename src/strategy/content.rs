//! Content-aware routing strategy.
//!
//! Routes based on message content analysis:
//! - Code/keywords → GPU (capable)
//! - Short simple queries → NPU (fast)
//! - Long context → GPU (big context window)
//! - Tool calls → GPU always

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// Content-aware router strategy.
pub struct ContentRouterStrategy {
    /// Default backend for small/simple requests.
    pub small_backend: String,

    /// Backend for complex/long requests.
    pub large_backend: String,

    /// Keywords that trigger large backend routing.
    pub gpu_keywords: Vec<String>,

    /// Max total tokens before switching to large backend permanently.
    pub max_small_tokens: usize,
}

impl ContentRouterStrategy {
    /// Check message content against GPU keywords.
    fn has_gpu_keywords(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.gpu_keywords.iter().any(|kw| lower.contains(&kw.to_lowercase()))
    }

    /// Estimate total input length from context.
    fn total_input_chars(&self, ctx: &Context) -> usize {
        ctx.messages.iter().map(|m| m.content.len()).sum()
    }
}

#[async_trait]
impl RouterStrategy for ContentRouterStrategy {
    fn name(&self) -> &'static str {
        "content_router"
    }

    async fn route(&self, ctx: &Context, _pool: &BackendPool) -> RoutingDecision {
        // Always switch to large backend after threshold
        if ctx.total_tokens >= self.max_small_tokens {
            return RoutingDecision::SingleToken {
                backend: self.large_backend.clone(),
            };
        }

        // Check the latest user message for GPU keywords
        if let Some(last_user_msg) = ctx.messages.iter().rev().find(|m| m.role == "user") {
            if self.has_gpu_keywords(&last_user_msg.content) {
                return RoutingDecision::SingleToken {
                    backend: self.large_backend.clone(),
                };
            }
        }

        // Long input → GPU
        if self.total_input_chars(ctx) > 800 {
            return RoutingDecision::SingleToken {
                backend: self.large_backend.clone(),
            };
        }

        // Default: fast small backend
        RoutingDecision::SingleToken {
            backend: self.small_backend.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, Message};
    use std::collections::HashMap;

    fn ctx_with_messages(texts: &[&str]) -> Context {
        let messages: Vec<Message> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let role = if i == 0 { "system".into() } else { "user".into() };
                Message {
                    role,
                    content: t.to_string(),
                }
            })
            .collect();
        Context {
            session_id: "test".into(),
            messages,
            generated: vec![],
            total_tokens: 0,
            max_tokens: 512,
            model_hint: None,
            stream: false,
        }
    }

    fn strategy() -> ContentRouterStrategy {
        ContentRouterStrategy {
            small_backend: "npu".into(),
            large_backend: "gpu".into(),
            gpu_keywords: vec![
                "code".into(), "explain".into(), "debug".into(),
                "refactor".into(), "write".into(), "fix".into(),
            ],
            max_small_tokens: 2000,
        }
    }

    #[tokio::test]
    async fn test_greeting_routes_npu() {
        let ctx = ctx_with_messages(&["You are helpful", "Hello!"]);
        let d = strategy().route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        assert!(matches!(d, RoutingDecision::SingleToken { backend } if backend == "npu"));
    }

    #[tokio::test]
    async fn test_code_routes_gpu() {
        let ctx = ctx_with_messages(&["You are a coder", "Write a Rust sort function"]);
        let d = strategy().route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        assert!(matches!(d, RoutingDecision::SingleToken { backend } if backend == "gpu"));
    }

    #[tokio::test]
    async fn test_debug_routes_gpu() {
        let ctx = ctx_with_messages(&["Assistant", "Help debug this Python crash on line 42"]);
        let d = strategy().route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        assert!(matches!(d, RoutingDecision::SingleToken { backend } if backend == "gpu"));
    }

    #[tokio::test]
    async fn test_long_context_routes_gpu() {
        let long = "a".repeat(1000);
        let ctx = ctx_with_messages(&["Assistant", &long]);
        let d = strategy().route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        assert!(matches!(d, RoutingDecision::SingleToken { backend } if backend == "gpu"));
    }
}
