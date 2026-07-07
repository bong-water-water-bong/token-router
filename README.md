# Token Router

**Token-level routing for multi-backend LLM inference.**

Route individual tokens to the optimal backend — NPU for fast cheap tokens, GPU for capable reasoning — all within a single streaming response.

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://rustup.rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)

## Why Token Routing?

Within a single response, some tokens are easy and some are hard:

| Token | Log-Prob | Route To |
|-------|----------|----------|
| "The" | -0.05 | **NPU** (fast, 291 tok/s) |
| "quantum" | -2.80 | **GPU** (capable, 113 tok/s) |
| "Fourier" | -0.10 | **NPU** (resume fast path) |

A **token-level router** makes this decision per-token, online. No existing open-source project does this.

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────┐
│   Client    │     │            Token Router                  │
│ (OpenAI SDK,│────▶│  /v1/chat/completions                    │
│  WebUI,     │◀────│  /v1/models  /v1/router  /v1/health     │
│  Continue)  │     └──────────┬──────────────────────────────┘
└─────────────┘                │
                               │ strategy: cascade / content / passthrough
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ NPU      │ │ GPU      │ │ MLX/vLLM │
             │ XDNA 2   │ │ Radeon   │ │ Cloud    │
             │ 291 t/s  │ │ 113 t/s  │ │ ...      │
             └──────────┘ └──────────┘ └──────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/bong-water-water-bong/token-router
cd token-router

# Build
cargo build --release

# Configure (edit router.toml to match your backends)
# Default: routes to http://127.0.0.1:13305

# Run
cargo run --release

# Or with custom config
cargo run --release -- --config my-router.toml
```

## Configuration

```toml
[server]
listen = "127.0.0.1:13306"
default_strategy = "cascade"

[backends]
  [backends.npu]
  type = "openai"
  base_url = "http://127.0.0.1:52628/v1"
  models = ["qwen3-0.6b-FLM"]
  speed_tok_s = 291

  [backends.gpu]
  type = "openai"
  base_url = "http://127.0.0.1:13305/v1"
  models = ["bonsai-1.7b"]
  speed_tok_s = 113

[strategies]
  [strategies.cascade]
  type = "cascade"
  small_backend = "npu"
  large_backend = "gpu"
  confidence_threshold = -2.5

  [strategies.content]
  type = "content_router"
  small_backend = "npu"
  large_backend = "gpu"
  gpu_keywords = ["code", "explain", "debug", "refactor"]

  [strategies.passthrough]
  type = "passthrough"
  backend = "gpu"
```

## Routing Strategies

### `cascade` ★ (Novel)
Token-level confidence routing. Start on NPU, check each token's log-prob,
seamlessly switch to GPU when confidence drops.

```
Input: "Explain the quantum Fourier transform"
  NPU: "Explain" (p=-0.1) ✓  →  "the" (p=-0.05) ✓
  NPU: "quantum"  (p=-2.8) ✗  →  SWITCH TO GPU
  GPU: "quantum Fourier transform"  →  [DONE]
```

**Config:**
| Field | Default | Description |
|-------|---------|-------------|
| `small_backend` | — | Fast backend (NPU) |
| `large_backend` | — | Capable backend (GPU) |
| `confidence_threshold` | -2.5 | Log-prob threshold for switching |
| `min_context_for_large` | 50 | Always use GPU after N tokens |

### `content_router`
Request-level routing by message analysis. Keywords, length, tool calls.

| Condition | Route |
|-----------|-------|
| `"Write code for..."` | GPU |
| `"Hello"` | NPU |
| Text > 800 chars | GPU |
| > 2000 tokens generated | GPU |

### `passthrough`
Simple proxy to a single backend.

### `spec_decode` (Scaffold)
Speculative decoding infrastructure. Draft on NPU, verify on GPU.

## API

The router exposes a standard OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with routing |
| `/v1/completions` | POST | Text completion (passthrough) |
| `/v1/models` | GET | Aggregated model list |
| `/v1/router` | GET | Router status and backend health |
| `/v1/router/metrics` | GET | Runtime metrics |
| `/health` | GET | Health check |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Route-Backend` | Which backend handled the request (`npu`, `gpu`, `cascade`) |

## Performance

On AMD Strix Halo (NPU + Radeon 8060S):

| Strategy | Est. Effective Speed | Use Case |
|----------|---------------------|----------|
| Passthrough (NPU) | 291 tok/s | Simple chat |
| Passthrough (GPU) | 113 tok/s | Complex reasoning |
| Cascade (threshold=-3) | ~200 tok/s | Mixed workloads |
| Content Router | — | Per-message routing |

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with verbose logging
RUST_LOG=debug cargo run

# Run with custom config
cargo run -- --config examples/npu-only.toml
```

## License

MIT
