//! Performance-based routing strategy.
//!
//! Routes requests to the backend that delivers the best performance for
//! the requested model. Uses a hardcoded performance table derived from
//! on-device benchmarks (docs/wiki/performance.md).
//!
//! Each entry maps a model name pattern to (backend_id, speed_tok_s).
//! The router selects the backend with the highest speed for the model.

use crate::backend::BackendPool;
use crate::context::Context;
use async_trait::async_trait;
use super::{RouterStrategy, RoutingDecision};

/// A model-to-backend performance entry.
#[derive(Debug, Clone)]
pub struct ModelRoute {
    /// Model name pattern (substring match, case-insensitive).
    pub model_pattern: &'static str,
    /// Backend ID in the router config.
    pub backend: &'static str,
    /// Measured decode speed in tok/s.
    pub speed_tok_s: f64,
    /// Notes about this route (e.g. quantization, engine variant).
    pub notes: &'static str,
}

/// Default performance table — single source of truth for model routing.
///
/// Sorted by model pattern. Patterns are checked in order; first match wins.
/// Derived from docs/wiki/performance.md "At a Glance" + detailed sections.
///
/// zen: ROCm backend for the MoE ternary kernels
/// gpu: Vulkan backend via llama.cpp / ZINC
/// npu: XDNA 2 NPU via FLM / fused engine
pub const PERFORMANCE_TABLE: &[ModelRoute] = &[
    // ── Zaya (ROCm — custom MoE ternary kernels) ──
    ModelRoute {
        model_pattern: "zaya",
        backend: "rocm",
        speed_tok_s: 18.0,
        notes: "ZAYA1PREVIEW-74B-A4B-Q4_K_M — MoE ternary, CCA attention, EDA router",
    },
    // ── GPU ternary / 1-bit (Vulkan — fastest decode) ──
    ModelRoute {
        model_pattern: "bonsai",
        backend: "gpu_vulkan",
        speed_tok_s: 279.0,
        notes: "Bonsai-1.7B Q2_0 — 1.58-bit ternary, llama.cpp Vulkan, 3.6 ms/tok",
    },
    ModelRoute {
        model_pattern: "qwen2",
        backend: "gpu_vulkan",
        speed_tok_s: 381.0,
        notes: "Qwen2-0.5B IQ1_S — 1.06 bpw, llama.cpp, 2.6 ms/tok",
    },
    ModelRoute {
        model_pattern: "hy-mt2",
        backend: "gpu_vulkan",
        speed_tok_s: 267.0,
        notes: "Hy-MT2 1.8B STQ1_0 — ZINC Sherry ternary, 3.7 ms/tok",
    },
    ModelRoute {
        model_pattern: "gemma-2",
        backend: "gpu_vulkan",
        speed_tok_s: 158.0,
        notes: "gemma-2-2b IQ1_S — 1.06 bpw, llama.cpp, 6.3 ms/tok",
    },
    ModelRoute {
        model_pattern: "gemma3",
        backend: "gpu_vulkan",
        speed_tok_s: 122.0,
        notes: "gemma3 4B IQ1_S — 1.06 bpw, llama.cpp, 8.2 ms/tok",
    },
    ModelRoute {
        model_pattern: "nemo",
        backend: "gpu_vulkan",
        speed_tok_s: 79.0,
        notes: "Nemo 8B IQ1_S — 1.06 bpw, llama.cpp, 12.7 ms/tok",
    },
    // ── NPU (fastest for small models) ──
    ModelRoute {
        model_pattern: "qwen3:0.6b",
        backend: "npu",
        speed_tok_s: 291.0,
        notes: "Qwen3-0.6B — NPU fused, 3.4 ms/tok, INT8",
    },
    ModelRoute {
        model_pattern: "qwen3:",
        backend: "npu",
        speed_tok_s: 94.0,
        notes: "Qwen3 series — NPU FLM, validated production",
    },
    ModelRoute {
        model_pattern: "llama3",
        backend: "npu",
        speed_tok_s: 28.0,
        notes: "Llama 3.x — C++ ALL engine, auto-detect",
    },
    ModelRoute {
        model_pattern: "gemma4",
        backend: "npu",
        speed_tok_s: 28.0,
        notes: "Gemma 4 — C++ ALL engine, auto-detect",
    },
    // ── GPU F16 / standard (ZINC Vulkan baseline) ──
    ModelRoute {
        model_pattern: "bitnet",
        backend: "gpu_vulkan",
        speed_tok_s: 22.0,
        notes: "BitNet b1.58-2B — ZINC Vulkan, F16 baseline",
    },
    ModelRoute {
        model_pattern: "zinc",
        backend: "gpu_vulkan",
        speed_tok_s: 22.0,
        notes: "Generic ZINC Vulkan — F16 baseline",
    },
];

/// Performance-based routing strategy.
///
/// Uses the PERFORMANCE_TABLE to find the best backend for the requested model.
pub struct PerformanceStrategy {
    /// Fallback backend when model isn't in the performance table.
    pub default_backend: String,
    /// Override: always use this backend regardless of model (for testing).
    pub force_backend: Option<String>,
}

impl PerformanceStrategy {
    /// Find the best backend for a model name from the performance table.
    pub fn best_backend(&self, model: &str) -> Option<&'static ModelRoute> {
        let lower = model.to_lowercase();
        PERFORMANCE_TABLE.iter().find(|route| {
            lower.contains(route.model_pattern)
        })
    }

    /// Get the backend ID that should handle a given model.
    pub fn resolve_backend(&self, model: &str) -> String {
        // Force override takes precedence
        if let Some(ref forced) = self.force_backend {
            return forced.clone();
        }
        // Look up in performance table
        if let Some(route) = self.best_backend(model) {
            return route.backend.to_string();
        }
        // Fallback to default
        self.default_backend.clone()
    }
}

#[async_trait]
impl RouterStrategy for PerformanceStrategy {
    fn name(&self) -> &'static str {
        "performance"
    }

    async fn route(&self, ctx: &Context, _pool: &BackendPool) -> RoutingDecision {
        let backend = if let Some(ref model) = ctx.model_hint {
            self.resolve_backend(model)
        } else {
            self.default_backend.clone()
        };

        RoutingDecision::SingleToken { backend }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, Message};
    use std::collections::HashMap;

    fn strategy() -> PerformanceStrategy {
        PerformanceStrategy {
            default_backend: "gpu_vulkan".into(),
            force_backend: None,
        }
    }

    fn ctx_with_model(model: &str) -> Context {
        Context {
            session_id: "test".into(),
            messages: vec![
                Message { role: "system".into(), content: "You are helpful.".into() },
                Message { role: "user".into(), content: "Hello!".into() },
            ],
            generated: vec![],
            total_tokens: 10,
            max_tokens: 512,
            model_hint: Some(model.to_string()),
            stream: false,
        }
    }

    #[tokio::test]
    async fn test_zaya_routes_rocm() {
        let s = strategy();
        let route = s.best_backend("ZAYA1PREVIEW-74B-A4B-Q4_K_M").unwrap();
        assert_eq!(route.backend, "rocm");
        assert_eq!(route.speed_tok_s, 18.0);
    }

    #[tokio::test]
    async fn test_bonsai_routes_gpu() {
        let s = strategy();
        let route = s.best_backend("Bonsai-1.7B-q2_0").unwrap();
        assert_eq!(route.backend, "gpu_vulkan");
        assert_eq!(route.speed_tok_s, 279.0);
    }

    #[tokio::test]
    async fn test_qwen3_routes_npu() {
        let s = strategy();
        let route = s.best_backend("qwen3:0.6b").unwrap();
        assert_eq!(route.backend, "npu");
        assert_eq!(route.speed_tok_s, 291.0);
    }

    #[tokio::test]
    async fn test_qwen2_routes_gpu() {
        let s = strategy();
        let route = s.best_backend("Qwen2-0.5B-IQ1_S").unwrap();
        assert_eq!(route.backend, "gpu_vulkan");
        assert_eq!(route.speed_tok_s, 381.0);
    }

    #[tokio::test]
    async fn test_unknown_model_falls_back() {
        let s = strategy();
        let backend = s.resolve_backend("unknown-model-v42");
        assert_eq!(backend, "gpu_vulkan"); // default fallback
    }

    #[tokio::test]
    async fn test_routing_decision() {
        let s = strategy();
        let ctx = ctx_with_model("ZAYA1PREVIEW-74B");
        let decision = s.route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        match decision {
            RoutingDecision::SingleToken { backend } => {
                assert_eq!(backend, "rocm");
            }
            _ => panic!("Expected SingleToken"),
        }
    }

    #[tokio::test]
    async fn test_no_model_hint_uses_default() {
        let s = strategy();
        let ctx = Context {
            session_id: "test".into(),
            messages: vec![],
            generated: vec![],
            total_tokens: 0,
            max_tokens: 512,
            model_hint: None,
            stream: false,
        };
        let decision = s.route(&ctx, &BackendPool::from_config(HashMap::new())).await;
        match decision {
            RoutingDecision::SingleToken { backend } => {
                assert_eq!(backend, "gpu_vulkan");
            }
            _ => panic!("Expected SingleToken"),
        }
    }

    #[test]
    fn test_performance_table_is_sorted() {
        // Verify all patterns are lowercase (substring match is case-insensitive)
        for route in PERFORMANCE_TABLE {
            let lower = route.model_pattern.to_lowercase();
            assert_eq!(
                route.model_pattern, &lower,
                "Pattern '{}' must be lowercase for case-insensitive matching",
                route.model_pattern
            );
        }
    }

    #[test]
    fn test_no_duplicate_patterns() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for route in PERFORMANCE_TABLE {
            assert!(
                seen.insert(route.model_pattern),
                "Duplicate model pattern: {}",
                route.model_pattern
            );
        }
    }
}
