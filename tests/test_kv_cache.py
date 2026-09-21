# Unit tests: ContiguousKVCache exactness + fp8-V attention equivalence.
import sys

import torch

from qwen_image_kernel.kv_cache import ContiguousKVCache
from qwen_image_kernel.triton_attention import block_causal_flash_attention, reference_block_causal_attention

torch.manual_seed(0)
DEV = "cuda"
B, P, Q, H, D = 1, 640, 256, 32, 128

ok = True

# --- 1) store/get round trip (bf16 V) ---
cache = ContiguousKVCache(2, B, P + 16, H, D, v_fp8=False, device=DEV)
k = torch.randn(B, P, H, D, device=DEV, dtype=torch.bfloat16)
v = torch.randn(B, P, H, D, device=DEV, dtype=torch.bfloat16)
cache.get_layer(0).store(k, v)
gk, gv = cache.get_layer(0).get()
d = max((gk - k).abs().max().item(), (gv - v).abs().max().item())
print(f"store/get bf16 roundtrip: {d:.6f} -> {'PASS' if d == 0 else 'FAIL'}")
ok &= d == 0

# 2) fp8-V store/get error bound (raw fp8 buffer + per-head scale == original)
v32 = v.float()
cache8 = ContiguousKVCache(2, B, P + 16, H, D, v_fp8=True, device=DEV)
cache8.get_layer(0).store(k, v)
gk8, gv8 = cache8.get_layer(0).get()
scale0 = cache8.v_scale[0]  # [B, H]
v_dequant = gv8.float() * scale0[:, :, None]  # per-head scale broadcast over (P, D)
rel = ((v_dequant - v.float()).norm() / v.float().norm()).item()
print(f"fp8-V store/get rel-err:  {rel:.5f} -> {'PASS' if rel < 0.03 else 'FAIL'}")
ok &= rel < 0.03

# 3) attention with fp8-V cache vs bf16-V reference
q = torch.randn(B, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
k_new = torch.randn(B, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
v_new = torch.randn(B, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
L = torch.full((Q,), P + Q - 1, dtype=torch.int32, device=DEV)

# bf16 cache reference
out_bf16 = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=k, cache_v=v)
ref = reference_block_causal_attention(q, k_new, v_new, L, None, cache_k=k, cache_v=v)
d_ref = (out_bf16.float() - ref.float()).abs().max().item()

# fp8 cache via the contiguous cache (store same k/v)
cache8.get_layer(1).store(k, v)
ck8, cv8, v_scale = cache8.get_layer(1).views()
out_fp8 = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck8, cache_v=cv8, cache_v_scale=v_scale)

d_fp8 = (out_fp8.float() - out_bf16.float()).abs().max().item()
rel_fp8 = ((out_fp8.float() - out_bf16.float()).norm() / out_bf16.float().norm()).item()
print(f"attn bf16-cache vs fp32 ref: {d_ref:.5f}")
print(f"attn fp8-V vs bf16-V: max {d_fp8:.5f} rel {rel_fp8:.5f} -> {'PASS' if rel_fp8 < 0.03 else 'FAIL'}")
ok &= rel_fp8 < 0.03

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)