# Definitive: same process, same tensors - wrapper vs direct kernel launch.
import sys, time

import torch
import triton

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from qwen_image_kernel.triton_attention import _qwen21_attn_fwd, block_causal_flash_attention

DEV = "cuda"
torch.manual_seed(0)
H, D = 32, 128
Q, P = 4096, 4123
KV = P + Q

q = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
k_new = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
v_new = torch.randn_like(k_new)
ck = torch.randn(1, P, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
cv = torch.randn_like(ck)
L = torch.full((Q,), KV - 1, dtype=torch.int32, device=DEV)
out = torch.empty_like(q)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

ms_wrap = bench(lambda: block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv))
print(f"wrapper:                   {ms_wrap:.2f} ms")

BM, BN, W, S = 128, 64, 4, 2
grid = (triton.cdiv(Q, BM), H)

def direct():
    _qwen21_attn_fwd[grid](
        q, ck, cv, k_new, v_new, out, L, q,
        128 ** -0.5, Q, Q, KV, H, P,
        q.stride(0), q.stride(1), q.stride(2),
        ck.stride(0), ck.stride(1), ck.stride(2),
        cv.stride(0), cv.stride(1), cv.stride(2),
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        0,
        HAS_CACHE=True, HAS_KEY_VALID=False,
        BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D,
        num_warps=W, num_stages=S,
    )

ms_direct = bench(direct)
print(f"direct (BM128,BN64,w4,s2): {ms_direct:.2f} ms")

grid2 = (triton.cdiv(Q, 128), H)

def direct2():
    _qwen21_attn_fwd[grid2](
        q, ck, cv, k_new, v_new, out, L, q,
        128 ** -0.5, Q, Q, KV, H, P,
        q.stride(0), q.stride(1), q.stride(2),
        ck.stride(0), ck.stride(1), ck.stride(2),
        cv.stride(0), cv.stride(1), cv.stride(2),
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        0,
        HAS_CACHE=True, HAS_KEY_VALID=False,
        BLOCK_M=128, BLOCK_N=128, HEAD_DIM=D,
        num_warps=8, num_stages=2,
    )

ms2 = bench(direct2)
print(f"direct (BM128,BN128,w8,s2): {ms2:.2f} ms")

o_w = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
direct()
o_d = out.clone()
print("wrapper output == direct output:", torch.equal(o_w, o_d))