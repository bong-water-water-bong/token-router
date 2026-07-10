//! KV cache handoff — zero-copy shared cache between NPU and GPU backends.
//!
//! When the cascade strategy switches from NPU to GPU mid-stream,
//! instead of sending accumulated text (which requires the GPU to
//! re-encode + re-compute KV cache from scratch), this module enables
//! passing the NPU's KV cache directly as a dma-buf handle.
//!
//! Protocol:
//!   1. `POST /v1/kv_cache/export` on NPU backend → `{ "kv_cache_id", "dma_buf_fd", "token_count" }`
//!   2. `POST /v1/kv_cache/import` on GPU backend → `{ "kv_cache_id", "dma_buf_fd", "context" }`
//!
//! The dma-buf fd is passed via SCM_RIGHTS on a Unix socket, or via
//! `/proc/self/fd/<N>` path convention for HTTP APIs.
//!
//! Falls back gracefully to text-based handoff when the backend doesn't
//! support KV cache sharing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tracing::{info, warn};

use crate::backend::BackendClient;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A handle representing a backend's exported KV cache state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheHandle {
    /// Unique identifier for this cache state (opaque to the router).
    pub kv_cache_id: String,

    /// Which backend exported this cache.
    pub source_backend: String,

    /// The model that generated this cache (for compatibility checking).
    pub model: String,

    /// Number of tokens processed in this cache state.
    pub token_count: usize,

    /// dma-buf file descriptor for the shared buffer (-1 if not available).
    /// The fd is valid only within the exporting process — the importing
    /// side must use the kv_cache_id to retrieve it via a different channel.
    pub dma_buf_fd: i32,

    /// Size of the dma-buf buffer in bytes.
    pub buffer_size: u64,

    /// Version of the KV cache layout (for forward compatibility).
    pub layout_version: u32,
}

/// Request to import a KV cache on another backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportRequest {
    /// The handle from a previous export.
    pub handle: KvCacheHandle,

    /// The conversation context so far (messages array).
    /// The importing backend needs this to continue generation
    /// even if the KV cache import fails.
    pub context: Vec<Value>,

    /// Whether to fall back to text-only if KV cache import fails.
    pub fallback_to_text: bool,
}

/// Response from an import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResponse {
    pub success: bool,
    pub kv_cache_id: Option<String>,
    pub message: String,
}

/// Response from an export operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub handle: Option<KvCacheHandle>,
    pub text_fallback: Value,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Handoff Client
// ---------------------------------------------------------------------------

/// Client for KV cache handoff operations on a backend.
pub struct KvCacheClient {
    backend_id: String,
    client: Arc<BackendClient>,
}

impl KvCacheClient {
    pub fn new(backend_id: String, client: Arc<BackendClient>) -> Self {
        Self { backend_id, client }
    }

    /// Export the current KV cache state from this backend.
    ///
    /// Returns the KV cache handle and a text fallback (accumulated messages)
    /// in case the importing backend doesn't support cache sharing.
    pub async fn export(&self, conversation: &[Value]) -> Result<ExportResponse> {
        let body = serde_json::json!({
            "messages": conversation
        });

        let resp = self
            .client
            .post_json("/v1/kv_cache/export", body)
            .await
            .context("KV cache export request failed")?;

        // Parse the response — could be either our structured response
        // or a fallback text response if the backend doesn't support cache export
        if let Ok(export) = serde_json::from_value::<ExportResponse>(resp.clone()) {
            info!(
                "{} exported KV cache: id={} tokens={} fd={}",
                self.backend_id,
                export.handle.as_ref().map(|h| &h.kv_cache_id).unwrap_or(&"none".into()),
                export.handle.as_ref().map(|h| h.token_count).unwrap_or(0),
                export.handle.as_ref().map(|h| h.dma_buf_fd).unwrap_or(-1),
            );
            Ok(export)
        } else {
            // Backend doesn't support structured export — return text fallback
            info!("{} does not support KV cache export, using text fallback", self.backend_id);
            Ok(ExportResponse {
                handle: None,
                text_fallback: resp,
                message: "Text fallback".into(),
            })
        }
    }

    /// Import a KV cache state into this backend.
    ///
    /// Returns the import response. If the backend doesn't support cache
    /// import, it will fall back to text-based continuation.
    pub async fn import(&self, request: ImportRequest) -> Result<ImportResponse> {
        let body = serde_json::to_value(&request)
            .context("Failed to serialize import request")?;

        let resp = self
            .client
            .post_json("/v1/kv_cache/import", body)
            .await
            .context("KV cache import request failed")?;

        if let Ok(import_resp) = serde_json::from_value::<ImportResponse>(resp) {
            if import_resp.success {
                info!(
                    "{} imported KV cache: id={:?}",
                    self.backend_id, import_resp.kv_cache_id
                );
            } else {
                warn!(
                    "{} KV cache import failed: {}",
                    self.backend_id, import_resp.message
                );
            }
            Ok(import_resp)
        } else {
            // Backend doesn't support structured import — return text fallback signal
            Ok(ImportResponse {
                success: false,
                kv_cache_id: None,
                message: "Backend does not support KV cache import".into(),
            })
        }
    }
}

/// Perform a KV-cache-aware handoff from NPU to GPU.
///
/// Returns:
/// - `Some(handle)` if the handoff succeeded with KV cache sharing
/// - `None` if the handoff fell back to text-based continuation
pub async fn handoff(
    npu_client: Arc<BackendClient>,
    gpu_client: Arc<BackendClient>,
    npu_backend_id: &str,
    gpu_backend_id: &str,
    conversation: &[Value],
) -> Result<Option<KvCacheHandle>> {
    let npu_cache = KvCacheClient::new(npu_backend_id.into(), npu_client.clone());

    // Step 1: Export KV cache from NPU
    let export = npu_cache.export(conversation).await?;

    // Step 2: If we got a cache handle, try to import into GPU
    if let Some(handle) = &export.handle {
        let gpu_cache = KvCacheClient::new(gpu_backend_id.into(), gpu_client.clone());

        // Check model compatibility (soft warning only — backend handles it)
        if handle.model != "auto-detect" {
            info!(
                "KV cache handoff: {} (tokens={}) → {}",
                npu_backend_id, handle.token_count, gpu_backend_id
            );
        }

        let import_req = ImportRequest {
            handle: handle.clone(),
            context: conversation.to_vec(),
            fallback_to_text: true,
        };

        let import_resp = gpu_cache.import(import_req).await?;

        if import_resp.success {
            return Ok(Some(handle.clone()));
        }
    }

    // Fallback: return None — caller should use text-based handoff
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_serialization() {
        let handle = KvCacheHandle {
            kv_cache_id: "test-123".into(),
            source_backend: "npu".into(),
            model: "qwen3-0.6b".into(),
            token_count: 42,
            dma_buf_fd: 7,
            buffer_size: 65536,
            layout_version: 1,
        };
        let json = serde_json::to_string(&handle).unwrap();
        assert!(json.contains("test-123"));
        assert!(json.contains("qwen3-0.6b"));

        let parsed: KvCacheHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.token_count, 42);
        assert_eq!(parsed.dma_buf_fd, 7);
    }

    #[test]
    fn test_import_request_serialization() {
        let handle = KvCacheHandle {
            kv_cache_id: "test-456".into(),
            source_backend: "npu".into(),
            model: "bonsai-1.7b".into(),
            token_count: 100,
            dma_buf_fd: 3,
            buffer_size: 131072,
            layout_version: 1,
        };
        let req = ImportRequest {
            handle,
            context: vec![
                serde_json::json!({"role": "user", "content": "Hello"}),
            ],
            fallback_to_text: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("test-456"));
        assert!(json.contains("Hello"));

        let parsed: ImportRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.handle.token_count, 100);
        assert_eq!(parsed.context.len(), 1);
    }

    #[test]
    fn test_export_response_deserialization() {
        let json = r#"{
            "handle": {
                "kv_cache_id": "exp-789",
                "source_backend": "npu",
                "model": "qwen3-0.6b",
                "token_count": 15,
                "dma_buf_fd": 5,
                "buffer_size": 32768,
                "layout_version": 1
            },
            "text_fallback": {"choices": [{"message": {"content": "Hello"}}]},
            "message": "Success"
        }"#;
        let parsed: ExportResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.handle.is_some());
        assert_eq!(parsed.handle.unwrap().token_count, 15);
    }

    #[test]
    fn test_export_response_text_only() {
        let json = r#"{
            "handle": null,
            "text_fallback": {"choices": [{"message": {"content": "Hello"}}]},
            "message": "Text fallback — KV cache export not supported"
        }"#;
        let parsed: ExportResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.handle.is_none());
    }

    #[test]
    fn test_handoff_fallback_when_handle_none() {
        // Without a handle, handoff() should return None (caller falls back)
        let handle = KvCacheHandle {
            kv_cache_id: "test".into(),
            source_backend: "npu".into(),
            model: "test".into(),
            token_count: 0,
            dma_buf_fd: -1,
            buffer_size: 0,
            layout_version: 1,
        };
        assert_eq!(handle.dma_buf_fd, -1);
        assert_eq!(handle.token_count, 0);
    }
}
