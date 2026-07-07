//! Speculative decode strategy.
//!
//! Draft N tokens from a fast backend (draft), then verify all N in one
//! forward pass on the capable backend (target). Accept tokens where
//! the target agrees with the draft above a threshold.
//!
//! This reuses the spec-decode pattern from `spec-decode/engine/spec_decode.h`

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// Speculative decoding strategy.
pub struct SpecDecodeStrategy {
    /// Fast draft backend (e.g., NPU).
    pub draft_backend: String,

    /// Target verification backend (e.g., GPU).
    pub target_backend: String,

    /// Number of draft tokens per round.
    pub n_draft: usize,

    /// Minimum acceptance probability.
    pub acceptance_threshold: f64,
}

#[async_trait]
impl RouterStrategy for SpecDecodeStrategy {
    fn name(&self) -> &'static str {
        "spec_decode"
    }

    async fn route(&self, _ctx: &Context, _pool: &BackendPool) -> RoutingDecision {
        RoutingDecision::Speculative {
            draft_backend: self.draft_backend.clone(),
            target_backend: self.target_backend.clone(),
            n_draft: self.n_draft,
        }
    }
}
