//! Routing strategies — the core abstraction.
//!
//! Each strategy implements `RouterStrategy` and decides which backend(s)
//! to use for each token, given the current context.

mod cascade;
mod content;
mod passthrough;
mod spec_decode;

pub use cascade::CascadeStrategy;
pub use content::ContentRouterStrategy;
pub use passthrough::PassthroughStrategy;
pub use spec_decode::SpecDecodeStrategy;

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;

/// The routing decision for the next token(s).
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Generate the next token from a single backend.
    SingleToken {
        backend: String,
    },

    /// Speculative: draft N tokens from one backend, verify on another.
    Speculative {
        #[allow(dead_code)]
        draft_backend: String,
        target_backend: String,
        #[allow(dead_code)]
        n_draft: usize,
    },
}

/// A routing strategy decides which backend to use for each generation step.
#[async_trait]
pub trait RouterStrategy: Send + Sync {
    /// Human-readable name for observability.
    fn name(&self) -> &'static str;

    /// Decide backend(s) for the next token given the current context.
    async fn route(&self, ctx: &Context, pool: &BackendPool) -> RoutingDecision;
}
