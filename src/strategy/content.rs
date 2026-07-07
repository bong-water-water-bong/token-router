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

    /// Max tokens before switching to large backend.
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
