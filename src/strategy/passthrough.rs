//! Passthrough strategy — forwards all requests to a single backend.
//! Used as default when no routing is needed.

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// Simple passthrough: all tokens go to one backend.
pub struct PassthroughStrategy {
    pub backend: String,
}

#[async_trait]
impl RouterStrategy for PassthroughStrategy {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    async fn route(&self, _ctx: &Context, _pool: &BackendPool) -> RoutingDecision {
        RoutingDecision::SingleToken {
            backend: self.backend.clone(),
        }
    }
}
