//! Configuration loader — parses `router.toml` into typed config structs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub backends: HashMap<String, BackendConfig>,
    pub strategies: HashMap<String, StrategyConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,

    #[serde(default = "default_log_level")]
    pub log_level: String,

    #[serde(default = "default_strategy")]
    pub default_strategy: String,
}

fn default_listen() -> String {
    "127.0.0.1:13306".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_strategy() -> String {
    "passthrough".to_string()
}

/// Backend connection details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: BackendType,

    pub base_url: String,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub models: Vec<String>,

    #[serde(default)]
    pub speed_tok_s: Option<f64>,

    #[serde(default = "default_cost")]
    pub cost_per_token: f64,
}

fn default_cost() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackendType {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "npu")]
    Npu,
    #[serde(rename = "rocm")]
    Rocm,
    #[serde(rename = "mlx")]
    Mlx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StrategyConfig {
    #[serde(rename = "passthrough")]
    Passthrough { backend: String },

    #[serde(rename = "cascade")]
    Cascade {
        small_backend: String,
        large_backend: String,
        #[serde(default = "default_confidence_threshold")]
        confidence_threshold: f64,
        #[serde(default = "default_min_context")]
        min_context_for_large: usize,
    },

    #[serde(rename = "spec_decode")]
    SpecDecode {
        draft_backend: String,
        target_backend: String,
        #[serde(default = "default_n_draft")]
        n_draft: usize,
        #[serde(default = "default_acceptance_threshold")]
        acceptance_threshold: f64,
    },

    #[serde(rename = "content_router")]
    ContentRouter {
        fallback_large_backend: String,
        #[serde(default)]
        gpu_keywords: Vec<String>,
        #[serde(default = "default_max_small_tokens")]
        max_small_tokens: usize,
    },
}

fn default_confidence_threshold() -> f64 {
    -2.5
}

fn default_min_context() -> usize {
    50
}

fn default_n_draft() -> usize {
    4
}

fn default_acceptance_threshold() -> f64 {
    0.8
}

fn default_max_small_tokens() -> usize {
    2000
}

impl Config {
    /// Load from a TOML file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to read config {}: {}", path.as_ref().display(), e))?;
        toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
    }

    /// Load from a string (for testing).
    #[allow(dead_code)]
    pub fn from_str(content: &str) -> anyhow::Result<Self> {
        toml::from_str(content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
    }

    /// Get the default config
    pub fn default_config() -> Self {
        Config {
            server: ServerConfig {
                listen: default_listen(),
                log_level: default_log_level(),
                default_strategy: default_strategy(),
            },
            backends: HashMap::new(),
            strategies: HashMap::new(),
        }
    }
}
