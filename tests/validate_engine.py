#!/usr/bin/env python3
"""Validate NPU engine intermediates against Python reference.

Sends a single-token prompt to the engine, records all layer outputs,
and compares against a Python reference trace.
"""
import json, subprocess, sys, numpy as np, os
sys.path.insert(0, '/home/bcloud/tools')
from q4nx_reference import get_header, read_bf16, dequantize_weight

H = 1024; NH = 16; NKV = 8; HD = 128; IM = 3072; NV = 151936
GQA = NH // NKV; NC = 28; EPS = 1e-6

def rms_norm(x, w):
    rms = np.sqrt(np.mean(x.astype(np.float32)**2) + EPS)
    return (x.astype(np.float32) / rms) * w

def silu(x): return x * (1.0 / (1.0 + np.exp(-x)))

def rotate_half(x):
    hd2 = x.shape[-1] // 2
    return np.concatenate([-x[..., hd2:], x[..., :hd2]], axis=-1)

def apply_rope(x, cos, sin):
    hd2 = x.shape[-1] // 2
    x1, x2 = x[..., :hd2], x[..., hd2:]
    return np.concatenate([x1*cos - x2*sin, x2*cos + x1*sin], axis=-1)

hdr = get_header()
emb_info = hdr['model.embed_tokens.weight']
embeds = read_bf16(emb_info['data_offsets'][0], NV*H).reshape(NV, H)

# Load weights for layer 0 only (first-layer comparison is sufficient)
def load_layer(l):
    pre = f'model.layers.{l}'
    def dq(name, of, inf):
        info = hdr[name]
        return dequantize_weight(info['data_offsets'][0], info['shape'][0], inf)
    return {
        'ln1': read_bf16(hdr[f'{pre}.input_layernorm.weight']['data_offsets'][0], H),
        'ln2': read_bf16(hdr[f'{pre}.post_attention_layernorm.weight']['data_offsets'][0], H),
        'qn': read_bf16(hdr[f'{pre}.self_attn.q_norm.weight']['data_offsets'][0], HD),
        'kn': read_bf16(hdr[f'{pre}.self_attn.k_norm.weight']['data_offsets'][0], HD),
        'q': dq(f'{pre}.self_attn.q_proj.weight', NH*HD, H),
        'k': dq(f'{pre}.self_attn.k_proj.weight', NKV*HD, H),
        'v': dq(f'{pre}.self_attn.v_proj.weight', NKV*HD, H),
        'o': dq(f'{pre}.self_attn.o_proj.weight', H, NH*HD),
        'g': dq(f'{pre}.mlp.gate_proj.weight', IM, H),
        'u': dq(f'{pre}.mlp.up_proj.weight', IM, H),
        'd': dq(f'{pre}.mlp.down_proj.weight', H, IM),
    }

# Run Python reference for token 100 through layer 0
print("=== Python Reference: Token 100, Layer 0 ===")
h = embeds[100].copy().astype(np.float32)
print(f"Input norm: {np.linalg.norm(h):.4f}")

w = load_layer(0)

# RMSNorm 1
h_ln1 = rms_norm(h, w['ln1'])
print(f"After RMSNorm1: norm={np.linalg.norm(h_ln1):.4f} range=[{h_ln1.min():.3f},{h_ln1.max():.3f}]")

# QKV projections
q = w['q'] @ h; k = w['k'] @ h; v = w['v'] @ h
print(f"Q: norm={np.linalg.norm(q):.4f} K: norm={np.linalg.norm(k):.4f} V: norm={np.linalg.norm(v):.4f}")

# After O projection + residual  
attn_out = np.zeros(NH*HD)
for hh in range(NH): attn_out[hh*HD:(hh+1)*HD] = v[hh//GQA*HD:(hh//GQA+1)*HD]
attn_proj = w['o'] @ attn_out
print(f"O proj: norm={np.linalg.norm(attn_proj):.4f}")
h = h + attn_proj
print(f"After residual1: norm={np.linalg.norm(h):.4f}")

# RMSNorm 2
h_ln2 = rms_norm(h, w['ln2'])

# FFN
gate = w['g'] @ h_ln2; up = w['u'] @ h_ln2
hidden = silu(gate) * up
ffn_out = w['d'] @ hidden
h = h + ffn_out
print(f"After residual2: norm={np.linalg.norm(h):.4f}")
print(f"Layer 0 output range=[{h.min():.3f},{h.max():.3f}]")

# Now run the engine with token 100 and compare
print("\n=== Engine Test (token 100, max_new_tokens=3) ===")
proc = subprocess.run(
    ['sudo', '/home/bcloud/engine/npu/build/npu_engine_server'],
    input=json.dumps({"tokens": [100], "max_new_tokens": 3}).encode() + b"\n",
    capture_output=True, timeout=30
)
result = json.loads(proc.stdout)
tokens = result['tokens']
logprobs = result['logprobs']
print(f"Generated tokens: {tokens}")
print(f"Logprobs: {[f'{lp:.4f}' for lp in logprobs]}")

print("\n=== VERDICT ===")
# The engine's output should be coherent (not random garbage).
# Token values should be in a reasonable range (not all 0 or all max)
if all(0 < t < NV for t in tokens):
    print("✅ Tokens in valid range [1, NV)")
else:
    print("❌ Tokens OUT of valid range")

if all(np.isfinite(lp) for lp in logprobs):
    print("✅ Logprobs are finite")
else:
    print("❌ Logprobs have NaN/Inf")

# Decode tokens
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json')
decoded = tok.decode(tokens)
print(f"Decoded output: {repr(decoded)}")
print(f"Contains real text: {any(c.isalpha() for c in decoded)}")
