//! Token Router — multi-backend token-level LLM router library.
//!
//! Sits between any OpenAI-compatible client and multiple inference backends
//! (NPU, GPU, MLX, vLLM), routing individual tokens based on strategy.

pub mod backend;
pub mod cascade;
pub mod config;
pub mod context;
pub mod handlers;
pub mod strategy;
pub mod kv_cache;
pub mod stream;

/// Version constant.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
