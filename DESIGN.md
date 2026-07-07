# Token Router — Design Sketch

A **token-level router** that sits between any OpenAI-compatible client and
multiple inference backends (NPU, GPU, MLX, vLLM), routing **individual tokens**
to the optimal backend based on confidence, latency, and capability.

## Why Token-Level Routing?

| Approach | What it does | Problem |
|----------|-------------|---------|
| Request router (existing) | Routes entire request to one backend | Wastes small-model efficiency on hard tokens |
| Speculative decode | Draft N, verify in batch | Draft+target must share architecture |
| **Token router** | Route *each token* to best backend | Novel — no existing open-source impl |

The insight: within a single response, some tokens are easy ("the", "is", "a")
and some are hard ("quantum", "Fourier", "recursively"). Easy tokens should
go to the fast NPU (291 tok/s). Hard tokens should go to the capable GPU (113 tok/s).
A token router makes this decision **per token, online**.

## Architecture

```
┌────────────┐     ┌────────────────────────────────────────────────────┐
│   Client   │     │              Token Router Daemon                   │
│ (OpenAI    │────▶│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  │
│  SDK /     │     │  │ OpenAI   │  │  Router      │  │  Streaming  │  │
│  WebUI)    │◀────│  │ API      │──│  Engine      │──│  Merger     │  │
└────────────┘     │  │ (axum)   │  │              │  │             │  │
                   │  └──────────┘  │  • Strategy   │  │  • Per-tok  │  │
                   │                │  • Confidence │  │    reorder  │  │
                   │                │  • Context    │  │  • SSE merge│  │
                   │                │  • Budget     │  └─────────────┘  │
                   │                └──────┬───────┘                    │
                   │                       │                            │
                   │                ┌──────▼───────┐                    │
                   │                │  Backend Pool                    │
                   │                │  (connection mgr, health, retry) │
                   │                └──────┬───────┘                    │
                   └───────────────────────┼────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
            ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
            │  NPU Engine   │     │  ROCm Engine  │      │  MLX / vLLM  │
            │  (XDNA 2)     │     │  (Radeon)     │      │  (macOS/GPU)  │
            │  Qwen3-0.6B   │     │  Bonsai-1.7B  │      │  Qwen3-8B     │
            │  ~291 tok/s   │     │  ~113 tok/s   │      │  ~40 tok/s    │
            └──────────────┘     └──────────────┘      └──────────────┘
```

## Core Abstraction: Router Strategy

```rust
/// A routing strategy decides which backend(s) to use for each token.
#[async_trait]
trait RouterStrategy: Send + Sync {
    /// Name for observability
    fn name(&self) -> &'static str;

    /// Decide backend for the next token given current context.
    /// Called BEFORE each token generation step.
    async fn route(&self, ctx: &Context) -> RoutingDecision;
}

struct Context {
    /// Full conversation so far (system prompt, messages, generated tokens)
    messages: Vec<Message>,

    /// Generated tokens so far in this response
    generated: Vec<GeneratedToken>,

    /// Log-probabilities for each generated token (if available)
    log_probs: Vec<Option<f32>>,

    /// Entropy of each generated token (uncertainty signal)
    entropies: Vec<Option<f32>>,

    /// Last hidden states from each backend (for speculative handoff)
    hidden_states: HashMap<BackendId, Vec<f32>>,

    /// Latency budget remaining (ms)
    latency_budget_ms: Option<f64>,

    /// User-specified model preference
    model_hint: Option<String>,
}

enum RoutingDecision {
    /// Generate next token from a single backend
    SingleToken { backend: BackendId },

    /// Generate N draft tokens from one backend, verify on another
    Speculative {
        draft_backend: BackendId,
        target_backend: BackendId,
        n_draft: usize,
    },

    /// Generate the same token from both backends, pick best
    Parallel { backends: Vec<BackendId>, selection: SelectionStrategy },

    /// Use previous token from this backend (cache hit)
    Cached { backend: BackendId },
}
```

## Built-in Strategies

### 1. `cascade` — Confidence-Based Cascade (MVP)
Try NPU first. If log-prob < threshold, reroute to GPU for that position.

```rust
// 1. NPU generates token at full speed (3.5ms/tok)
// 2. If log_prob(token) < threshold → GPU regenerates this position
// 3. GPU output replaces NPU output, stream continues from GPU state
// 4. NPU catches up on next easy token
//
// Threshold tuning:
//   -1.0 = very aggressive (routes ~50% to GPU)
//   -3.0 = conservative (routes ~10% to GPU)
//   -5.0 = only for very uncertain tokens
```

**Benchmark estimate:**
| Threshold | NPU tokens | GPU tokens | Est. speed |
|-----------|-----------|------------|------------|
| `always_npu` | 100% | 0% | 291 tok/s |
| `-5.0` | ~95% | ~5% | ~230 tok/s |
| `-3.0` | ~85% | ~15% | ~160 tok/s |
| `-1.0` | ~60% | ~40% | ~85 tok/s |
| `always_gpu` | 0% | 100% | 113 tok/s |

### 2. `spec_decode` — Speculative Decode (Use Existing)
Reuse the `spec-decode/` engine: draft from NPU (fast), verify on GPU (capable).

```rust
// 1. NPU generates N=4 draft tokens greedily
// 2. GPU verifies all N in one forward pass
// 3. Accept tokens where GPU logits agree with draft
// 4. Continue from the last accepted token
```

**Why this works on Strix Halo:** Unified memory means NPU → GPU handoff
is zero-copy. No PCIe bottleneck.

### 3. `content_router` — Content-Aware Routing (Simple)
Route based on message content. Upgrade of the existing `unified-router.py`.

```rust
// Keywords → GPU: "code", "debug", "refactor", "explain", "analyze"
// Short text → NPU: "hi", "yes", "what's 2+2?"
// Tool calls → GPU always
// Long context → GPU always
```

### 4. `cost_optimal` — Budget-Aware Routing
Given a latency budget (e.g., "never exceed 50ms/tok"), pick the fastest
backend that can meet it, falling back to slower but more capable backends
only when confidence drops.

## Backend Pool

```rust
struct BackendConfig {
    id: BackendId,
    label: String,
    backend_type: BackendType,
    base_url: String,
    api_key: Option<String>,
    models: Vec<String>,
    capabilities: Vec<Capability>,
    speed_tok_s: f64,       // measured or advertised
    cost_per_token: f64,    // relative cost metric
}

enum BackendType {
    /// OpenAI-compatible HTTP API
    OpenAI,
    /// NPU engine via UNIX pipe or TCP (custom binary protocol)
    NpuEngine,
    /// ROCm engine via subprocess (spawns bitnet_decode)
    RocmSubprocess,
    /// MLX engine (macOS)
    Mlx,
}

enum Capability {
    Streaming,
    ToolCalls,
    Vision,
    FunctionCalling,
    LongContext(usize),  // max context length
}
```

## Data Flow: Streaming with Token Handoff

```
Client                Token Router              NPU                  GPU
  │                        │                    │                    │
  │  POST /v1/chat/        │                    │                    │
  │  completions           │                    │                    │
  │───────────────────────▶│                    │                    │
  │                        │                    │                    │
  │                        │  Route: NPU first  │                    │
  │                        │──────────────────▶│                    │
  │                        │                    │                    │
  │                        │  "The" (p=0.98)    │                    │
  │  ◀── SSE: "The" ──────│◀───────────────────│                    │
  │                        │                    │                    │
  │                        │  "quantum" (p=0.12) │                    │
  │                        │◀───────────────────│                    │
  │                        │                    │                    │
  │                        │  ┌── LOW CONFIDENCE                   │
  │                        │  │  Route this token to GPU           │
  │                        │  └───────────────────────────────────▶│
  │                        │                    │                    │
  │                        │                    │  "quantum" (p=0.89)│
  │                        │  ◀────────────────────────────────────│
  │                        │                    │                    │
  │  ◀── SSE: "quantum" ──│  (GPU output replaces NPU)             │
  │                        │                    │                    │
  │                        │  Route: NPU resume │                    │
  │                        │──────────────────▶│                    │
  │                        │  "Fourier" (p=0.94)│                    │
  │  ◀── SSE: "Fourier" ──│◀───────────────────│                    │
```

## Configuration File (`router.toml`)

```toml
[server]
listen = "127.0.0.1:13306"     # Router port (different from backends)
log_level = "info"
default_strategy = "cascade"

[backends]
  [backends.npu]
  type = "openai"
  base_url = "http://127.0.0.1:52628/v1"
  models = ["qwen3-0.6b-FLM"]
  speed_tok_s = 291
  cost_per_token = 0.1

  [backends.gpu]
  type = "openai"
  base_url = "http://127.0.0.1:13305/v1"
  models = ["bonsai-1.7b", "bitnet"]
  speed_tok_s = 113
  cost_per_token = 1.0

  [backends.mlx]
  type = "openai"
  base_url = "http://127.0.0.1:8080/v1"
  models = ["mlx-community/Qwen3-8B-4bit"]
  speed_tok_s = 40
  cost_per_token = 2.0

[strategies.cascade]
type = "cascade"
small_backend = "npu"
large_backend = "gpu"
confidence_threshold = -2.5     # log-prob threshold
min_context_for_large = 50       # always use GPU after 50 tokens

[strategies.spec_decode]
type = "spec_decode"
draft_backend = "npu"
target_backend = "gpu"
n_draft = 4
acceptance_threshold = 0.8       # min probability for acceptance

[strategies.content]
type = "content_router"
fallback_large_backend = "gpu"
# Keywords that trigger GPU routing
gpu_keywords = [
  "code", "explain", "analyze", "debug", "refactor",
  "implement", "function", "algorithm", "architecture"
]
max_small_tokens = 2000          # switch to GPU after N tokens
```

## Project Structure

```
token-router/
├── Cargo.toml                  # Rust project (axum, tokio, serde, tower)
├── router.toml                 # Default config
├── DESIGN.md                   # This document
├── README.md                   # Quick start, install
├── LICENSE                     # MIT
│
├── src/
│   ├── main.rs                 # Entry point, server bootstrap
│   ├── config.rs               # TOML config parser
│   ├── server.rs               # Axum HTTP server, OpenAI API routes
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── pool.rs             # Backend connection pool, health checks
│   │   ├── client.rs           # Generic HTTP client for OpenAI-compatible backends
│   │   └── npu_pipe.rs         # NPU engine direct pipe (optional, zero-copy)
│   │
│   ├── strategy/
│   │   ├── mod.rs              # RouterStrategy trait
│   │   ├── cascade.rs          # Confidence-based cascade
│   │   ├── spec_decode.rs      # Speculative decode
│   │   ├── content.rs          # Content-aware routing
│   │   └── cost_optimal.rs     # Budget-aware routing
│   │
│   ├── context.rs              # Context manager (message history, KV state)
│   ├── stream.rs               # Streaming response merger, SSE
│   ├── confidence.rs           # Confidence scoring, entropy calculation
│   └── metrics.rs              # Prometheus metrics, per-backend stats
│
├── tests/
│   ├── integration_test.rs     # Multi-backend integration tests
│   └── strategy_tests.rs       # Unit tests for each strategy
│
└── examples/
    ├── docker-compose.yml      # NPU + GPU + Router all-in-one
    └── open-webui-config.json  # How to point Open WebUI at the router
```

## Key Design Decisions

### 1. Rust + Axum
- Matches existing `onebit` server pattern — users already have the stack
- Async streaming (SSE) is first-class
- `reqwest` for backend proxying
- `tokio` for async subprocess management (NPU engine, bitnet_decode)

### 2. OpenAI-Compatible API
- The router exposes the **exact** `POST /v1/chat/completions` API
- Any OpenAI-compatible client works unchanged (Open WebUI, Continue, Cline, Aider)
- Passes through models list from all backends aggregated

### 3. Streaming-First
- All strategies support SSE streaming
- Token handoff (NPU→GPU mid-stream) is seamless — client sees one stream
- Use `model` parameter to pin a strategy: `model = "cascade"`, `model = "npu+fast"`

### 4. Pluggable Strategies
- `RouterStrategy` trait is the extension point
- Anyone can add a new strategy without changing core
- Strategies compose: `content_router → cascade → spec_decode`

### 5. No Python in the Hot Path
- Router is pure Rust, zero Python dependencies
- Backend communication is standard HTTP (OpenAI API) or subprocess pipe
- Python tools (training, eval) are build-time only

## MVP Milestones

### Phase 1: Backend Pool + Passthrough (~1 day)
- [ ] Config loader (TOML)
- [ ] Backend pool: HTTP clients, health checks, model listing
- [ ] Passthrough router: forward all requests to one backend
- [ ] OpenAI API endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`

### Phase 2: Content Router (~4 hours)
- [ ] Port `unified-router.py` keyword-based routing to Rust
- [ ] Streaming passthrough with correct SSE framing

### Phase 3: Cascade Strategy (core innovation, ~2 days)
- [ ] NPU request/response with logprobs
- [ ] Confidence threshold detection
- [ ] GPU handoff at specific token position
- [ ] Streaming merger (interleave NPU/GPU outputs seamlessly)

### Phase 4: Speculative Decode (~3 days)
- [ ] Integrate with `spec-decode/engine/spec_decode.h`
- [ ] Draft on NPU, verify on GPU
- [ ] Token acceptance logic

### Phase 5: Polish (~1 day)
- [ ] Prometheus metrics: per-backend latency, token counts, routing decisions
- [ ] Graceful shutdown, hot-reload config
- [ ] Docker images for common setups

## Why This Is Worth Open-Sourcing

1. **No equivalent exists** — there are request routers (LiteLLM, OpenRouter) but
   no open-source *token-level* routers that can switch backends mid-stream

2. **Strix Halo is unique hardware** — the NPU+GPU unified memory makes
   zero-copy token handoff possible. Nobody else is building for this.

3. **Solves a real problem** — every inference deployment faces the
   "small vs large model" tradeoff. A token router makes it dynamic.

4. **Vendor-agnostic** — works with any OpenAI-compatible backend.
   NPU, GPU, MLX, vLLM, Ollama, even cloud APIs. The Strix Halo optimizations
   are just the killer demo.

5. **Composability** — cascade + content routing + speculation can all
   be active simultaneously. Each token gets the optimal treatment.
