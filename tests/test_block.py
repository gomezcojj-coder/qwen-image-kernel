# Integration test: QwenImage21KernelBlock vs the stock block, end to end
# (prefill + KV-cache decode), on a miniature text+image joint sequence.
import sys

import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21TransformerBlock,
    _qwenimage21_prefix_segments,
)

from qwen_image_kernel.processors import (
    QwenImage21KernelBlock,
    QwenImage21KernelProcessor,
    _build_prefix_lengths,
)

torch.manual_seed(0)
DEV = "cuda"

DIM = 256          # heads * head_dim
HEADS = 4
HEAD_DIM = 64

# joint layout: text(37) -> image block(16) -> text(11) -> target(25)
layout = [("text", 37), ("img", 16), ("text", 11), ("img", 25)]
segments = []
pos = 0
image_ids = []
for i, (kind, n) in enumerate(layout):
    segments.append((pos, pos + n, kind == "text"))
    image_ids.extend([-1] * n if kind == "text" else [i] * n)
    pos += n
S = pos
prefix_len = segments[-2][1]   # target = last segment
S_target = layout[-1][1]
assert prefix_len == 64

B = 1
torch.manual_seed(42)
stock = QwenImage21TransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, mlp_ratio=3, eps=1e-6).to(DEV, torch.bfloat16).eval()
kernel = QwenImage21KernelBlock(stock).to(DEV).eval()
kernel.attn.set_processor(QwenImage21KernelProcessor())

hidden = torch.randn(B, S, DIM, device=DEV, dtype=torch.bfloat16)
modulation = torch.randn(B + 1, 4 * DIM, device=DEV, dtype=torch.bfloat16) * 0.2
target_token_mask = torch.zeros(S, dtype=torch.bool, device=DEV)
target_token_mask[prefix_len:] = True

inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device=DEV).float() / HEAD_DIM))
ang = torch.outer(torch.arange(S, device=DEV).float(), inv)
freqs = torch.polar(torch.ones_like(ang), ang)  # [S, D/2] complex

common = dict(rotary_emb=freqs, target_token_mask=target_token_mask)

# ---------------- prefill (no cache) ----------------
with torch.no_grad():
    ref = stock(
        hidden_states=hidden.clone(),
        modulation=modulation,
        rotary_emb=freqs,
        target_token_mask=target_token_mask,
        segments=segments,
        key_valid=None,
    )
    kb = QwenImage21KVCache(1)
    L = _build_prefix_lengths(segments, prefix_len, S, DEV)
    _ = L  # kernel processor builds its own from segments
    mine = kernel(
        hidden_states=hidden.clone(),
        modulation=modulation,
        rotary_emb=freqs,
        target_token_mask=target_token_mask,
        segments=segments,
        key_valid=None,
    )
d = (mine.float() - ref.float()).abs().max().item()
print(f"block prefill        max|diff| {d:.6f} -> {'PASS' if d < 4e-2 else 'FAIL'}")
ok = d < 4e-2

# ---------------- prefill with cache extraction ----------------
with torch.no_grad():
    ref_cache = QwenImage21KVCache(1)
    ref2 = stock(
        hidden_states=hidden.clone(),
        modulation=modulation,
        rotary_emb=freqs,
        target_token_mask=target_token_mask,
        layer_cache=ref_cache.get_layer(0),
        kv_cache_mode="extract",
        cache_write_slice=slice(0, prefix_len),
        segments=segments,
        key_valid=None,
    )
    d2 = (ref2.float() - ref.float()).abs().max().item()
    print(f"  (stock cache-extract vs plain prefill diff {d2:.5f})")

    mine_cache = QwenImage21KVCache(1)
    mine2 = kernel(
        hidden_states=hidden.clone(),
        modulation=modulation,
        rotary_emb=freqs,
        target_token_mask=target_token_mask,
        layer_cache=mine_cache.get_layer(0),
        kv_cache_mode="extract",
        cache_write_slice=slice(0, prefix_len),
        segments=segments,
        key_valid=None,
    )
d3 = (mine2.float() - ref2.float()).abs().max().item()
print(f"block prefill+extract max|diff| {d3:.6f} -> {'PASS' if d3 < 4e-2 else 'FAIL'}")
ok &= d3 < 4e-2

# cached K/V match?
rk, rv = ref_cache.get_layer(0).get()
mk, mv = mine_cache.get_layer(0).get()
dc = max((mk.float() - rk.float()).abs().max().item(), (mv.float() - rv.float()).abs().max().item())
print(f"  cached K/V         max|diff| {dc:.6f} -> {'PASS' if dc < 4e-2 else 'FAIL'}")
ok &= dc < 4e-2

# ---------------- decode from cache ----------------
h_target = hidden[:, prefix_len:].contiguous()
all_target = torch.ones(S_target, dtype=torch.bool, device=DEV)  # decode rows are all target rows
with torch.no_grad():
    ref_dec = stock(
        hidden_states=h_target.clone(),
        modulation=modulation,
        rotary_emb=freqs[prefix_len:],
        attention_mask=None,
        target_token_mask=all_target,
        layer_cache=ref_cache.get_layer(0),
        kv_cache_mode="cached",
    )
    mine_dec = kernel(
        hidden_states=h_target,
        modulation=modulation,
        rotary_emb=freqs[prefix_len:],
        attention_mask=None,
        target_token_mask=all_target,
        layer_cache=mine_cache.get_layer(0),
        kv_cache_mode="cached",
    )
d4 = (mine_dec.float() - ref_dec.float()).abs().max().item()
print(f"block decode+cache   max|diff| {d4:.6f} -> {'PASS' if d4 < 4e-2 else 'FAIL'}")
ok &= d4 < 4e-2

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)