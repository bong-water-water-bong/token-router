//! Speculative decode strategy.
//!
//! Draft N tokens from a fast backend (draft), then verify all N in one
//! forward pass on the capable backend (target). Accept tokens where
//! the target agrees with the draft using rejection sampling.
//!
//! The actual streaming engine lives in `crate::spec_decode::spec_decode_stream`.
//! This module provides the strategy decision and config.

use crate::backend::BackendPool;
use crate::context::Context;
use crate::spec_decode;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// Speculative decoding strategy.
pub struct SpecDecodeStrategy {
    /// Fast draft backend (e.g., NPU Qwen3-0.6B).
    pub draft_backend: String,

    /// Target verification backend (e.g., GPU Bonsai-1.7B).
    pub target_backend: String,

    /// Number of draft tokens per round (default: 4).
    pub n_draft: usize,

    /// Minimum acceptance probability for rejection sampling (default: 0.8).
    pub acceptance_threshold: f64,

    /// Whether to enable dynamic n_draft adjustment based on acceptance rate.
    pub dynamic_n_draft: bool,
}

impl SpecDecodeStrategy {
    /// Create a new speculative decode strategy with default settings.
    pub fn new(draft_backend: String, target_backend: String) -> Self {
        Self {
            draft_backend,
            target_backend,
            n_draft: 4,
            acceptance_threshold: 0.8,
            dynamic_n_draft: true,
        }
    }

    /// Get speculative decode metrics from the engine.
    pub fn engine_metrics(&self) -> serde_json::Value {
        spec_decode::metrics()
    }
}

#[async_trait]
impl RouterStrategy for SpecDecodeStrategy {
    fn name(&self) -> &'static str {
        "spec_decode"
    }

    async fn route(&self, ctx: &Context, pool: &BackendPool) -> RoutingDecision {
        // Check that both backends exist
        let has_draft = pool.client(&self.draft_backend).is_some();
        let has_target = pool.client(&self.target_backend).is_some();

        if !has_draft || !has_target {
            // Fall back to single backend if either is unavailable
            let fallback: String = if has_target {
                self.target_backend.clone()
            } else if has_draft {
                self.draft_backend.clone()
            } else {
                // No backends available — return first configured
                pool.backend_ids().first().cloned().unwrap_or_default()
            };
            return RoutingDecision::SingleToken { backend: fallback };
        }

        // For very short sequences, spec decode overhead isn't worth it
        if ctx.total_tokens < 4 {
            return RoutingDecision::SingleToken {
                backend: self.draft_backend.clone(),
            };
        }

        RoutingDecision::Speculative {
            draft_backend: self.draft_backend.clone(),
            target_backend: self.target_backend.clone(),
            n_draft: self.n_draft,
        }
    }
}
