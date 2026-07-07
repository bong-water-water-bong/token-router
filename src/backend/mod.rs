//! Backend pool — manages connections to all inference backends.

mod client;

pub use client::BackendClient;

use crate::config::BackendConfig;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Unique identifier for a backend instance.
pub type BackendId = String;

/// Capabilities a backend may advertise or support.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Capability {
    Streaming,
    ToolCalls,
    Vision,
    FunctionCalling,
    LongContext(usize),
}

/// Runtime state for a single backend.
#[derive(Debug, Clone)]
pub struct BackendState {
    pub id: BackendId,
    pub config: BackendConfig,
    pub healthy: bool,
    pub last_checked: Instant,
    #[allow(dead_code)]
    pub latency_p50_ms: f64,
    pub latency_ema_ms: f64,
    pub circuit_open: bool,
}

/// The backend pool manages connections to all configured backends.
pub struct BackendPool {
    backends: DashMap<BackendId, BackendState>,
    clients: DashMap<BackendId, Arc<BackendClient>>,
}

impl BackendPool {
    /// Create a pool from the configuration.
    pub fn from_config(configs: impl IntoIterator<Item = (String, BackendConfig)>) -> Self {
        let pool = Self {
            backends: DashMap::new(),
            clients: DashMap::new(),
        };

        for (id, cfg) in configs {
            let initial_latency = 1000.0 / cfg.speed_tok_s.unwrap_or(100.0);
            let client = Arc::new(BackendClient::new(&cfg.base_url, cfg.api_key.as_deref()));
            pool.clients.insert(id.clone(), client);
            pool.backends.insert(id.clone(), BackendState {
                id: id.clone(),
                healthy: true,
                last_checked: Instant::now(),
                latency_p50_ms: initial_latency,
                latency_ema_ms: initial_latency,
                circuit_open: false,
                config: cfg,
            });
            info!(backend = %id, "Registered backend");
        }

        pool
    }

    /// Get a client by backend ID.
    pub fn client(&self, id: &str) -> Option<Arc<BackendClient>> {
        self.clients.get(id).map(|c| c.clone())
    }

    /// Get backend state.
    #[allow(dead_code)]
    pub fn state(&self, id: &str) -> Option<BackendState> {
        self.backends.get(id).map(|s| s.clone())
    }

    /// List all registered backend IDs.
    pub fn backend_ids(&self) -> Vec<String> {
        self.backends.iter().map(|e| e.key().clone()).collect()
    }

    /// List all backend states.
    pub fn all_states(&self) -> Vec<BackendState> {
        self.backends.iter().map(|e| e.value().clone()).collect()
    }

    /// Health-check a specific backend.
    pub async fn health_check(&self, id: &str) -> bool {
        let client = match self.clients.get(id) {
            Some(c) => c.clone(),
            None => return false,
        };

        match client.health_check().await {
            Ok(_) => {
                if let Some(mut state) = self.backends.get_mut(id) {
                    state.healthy = true;
                    state.last_checked = Instant::now();
                }
                true
            }
            Err(e) => {
                warn!(backend = %id, error = %e, "Health check failed");
                if let Some(mut state) = self.backends.get_mut(id) {
                    state.healthy = false;
                    state.last_checked = Instant::now();
                }
                false
            }
        }
    }

    /// Health-check all backends concurrently.
    pub async fn health_check_all(&self) {
        let ids: Vec<String> = self.backend_ids();
        let mut handles = Vec::new();
        for id in &ids {
            handles.push(self.health_check(id));
        }
        futures::future::join_all(handles).await;
    }

    /// Refresh latency EMA and circuit breaker state from live client metrics.
    pub async fn refresh_metrics(&self) {
        for mut entry in self.backends.iter_mut() {
            let id = entry.key().clone();
            if let Some(client) = self.clients.get(&id) {
                let state = entry.value_mut();
                state.latency_ema_ms = client.latency.current().await;
                state.circuit_open = !client.is_available();
            }
        }
    }
}
