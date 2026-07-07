//! Cascade strategy — starts with a fast small model (NPU), and routes
//! low-confidence tokens to a larger model (GPU).
//!
//! The core insight: within a single response, some tokens are easy
//! ("the", "is", "a") and some are hard ("quantum", "recursively").
//! Easy tokens go to the fast NPU; hard tokens go to the capable GPU.

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// Cascade routing strategy.
pub struct CascadeStrategy {
    /// Fast, cheap backend (e.g., NPU Qwen3-0.6B).
    pub small_backend: String,

    /// Capable, slower backend (e.g., GPU Bonsai-1.7B).
    pub large_backend: String,

    /// Log-probability threshold. Below this → route to large backend.
    /// Typical range: -1.0 (aggressive) to -5.0 (conservative).
    pub confidence_threshold: f64,

    /// After N tokens, always use large backend (avoids degenerate routing).
    pub min_context_for_large: usize,
}

/// The cascade can be in one of two states per-token
#[allow(dead_code)]
enum CascadeState {
    /// Using the small backend normally
    Small,
    /// Switched to large backend for this position
    Large,
}

#[async_trait]
impl RouterStrategy for CascadeStrategy {
    fn name(&self) -> &'static str {
        "cascade"
    }

    async fn route(&self, ctx: &Context, _pool: &BackendPool) -> RoutingDecision {
        // If we've generated enough tokens, switch to large backend permanently
        if ctx.total_tokens >= self.min_context_for_large {
            return RoutingDecision::SingleToken {
                backend: self.large_backend.clone(),
            };
        }

        // Check confidence of the last token
        if let Some(log_prob) = ctx.last_log_prob() {
            if (log_prob as f64) < self.confidence_threshold {
                // Low confidence → route this position to large backend
                return RoutingDecision::SingleToken {
                    backend: self.large_backend.clone(),
                };
            }
        }

        // Default: use the fast small backend
        RoutingDecision::SingleToken {
            backend: self.small_backend.clone(),
        }
    }
}
