# Sweep attention kernel configs at edit-decode shapes to find the fast one.
import sys, time

import torch
import triton

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)
from qwen_image_kernel.triton_attention import _qwen21_attn_fwd, block_causal_flash_attention

DEV = "cuda"
torch.manual_seed(0)

def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

# shapes: edit decode (4096 q, 4537 prefix + 4096 fresh)
Q, P, SN, H, D = 4096, 4537, 4096, 32, 128
KV = P + Q
q = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
k_new = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
v_new = torch.randn_like(k_new)
ck = torch.randn(1, P, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
cv = torch.randn_like(ck)
L = torch.full((Q,), KV - 1, dtype=torch.int32, device=DEV)
out = torch.empty_like(q)

flops = 2 * 2 * Q * KV * D * H  # QK + PV

def run(BM, BN, warps, stages):
    grid = (triton.cdiv(Q, BM), H)
    def call():
        _qwen21_attn_fwd[grid](
            q, ck, cv, k_new, v_new, out, L, q,
            q.shape[-1] ** -0.5, Q, Q, KV, H, P,
            q.stride(0), q.stride(1), q.stride(2),
            ck.stride(0), ck.stride(1), ck.stride(2),
            cv.stride(0), cv.stride(1), cv.stride(2),
            k_new.stride(0), k_new.stride(1), k_new.stride(2),
            v_new.stride(0), v_new.stride(1), v_new.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            0,
            HAS_CACHE=True, HAS_KEY_VALID=False,
            BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D,
            num_warps=warps, num_stages=stages,
        )
    ms = bench(call)
    print(f"BM={BM:3d} BN={BN:3d} warps={warps} stages={stages}: {ms:8.2f} ms  ({flops / ms / 1e9:6.1f} TFLOPS)")

for bm, bn, w, s in [
    (128, 64, 4, 2),   # current
    (128, 128, 4, 2),
    (128, 128, 8, 3),
    (128, 128, 8, 2),
    (128, 64, 8, 3),
    (64, 128, 4, 3),
    (64, 128, 8, 3),
    (256, 128, 8, 2),
    (128, 256, 8, 2),
    (64, 64, 4, 3),
]:
    try:
        run(bm, bn, w, s)
    except Exception as e:
        print(f"BM={bm} BN={bn} warps={w} stages={s}: FAIL {str(e)[:80]}")

# SDPA comparison on the same (dense) workload
import torch.nn.functional as F
qc = q.transpose(1, 2).contiguous()
kc = torch.cat([ck, k_new], 1).transpose(1, 2).contiguous()
vc = torch.cat([cv, v_new], 1).transpose(1, 2).contiguous()
def sdpa():
    return F.scaled_dot_product_attention(qc, kc, vc)
ms = bench(sdpa)
print(f"SDPA (flash) same dense workload: {ms:.2f} ms ({flops / ms / 1e9:.1f} TFLOPS)")