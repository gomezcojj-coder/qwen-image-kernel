# Confirm the WDDM spill hypothesis: bench attention with vs without the model resident.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from qwen_image_kernel.triton_attention import block_causal_flash_attention

DEV = "cuda"
torch.manual_seed(0)
H, D = 32, 128
Q, P = 4096, 4123

q = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
k_new = torch.randn(1, Q, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
v_new = torch.randn_like(k_new)
ck = torch.randn(1, P, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
cv = torch.randn_like(ck)
L = torch.full((Q,), P + Q - 1, dtype=torch.int32, device=DEV)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"{tag:32s} free {free / 2**30:5.2f} / {total / 2**30:.2f} GiB", flush=True)

def bench_attn():
    ms = bench(lambda: block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv))
    print(f"  attention: {ms:.2f} ms/call")

# 1) clean process
vram("before model")
bench_attn()

# 2) with a 7.4 GB model resident (simulate the edit path)
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
rt.transformer = rt._build_transformer()
vram("with model resident")
bench_attn()

# 3) free the model
rt.transformer.to("cpu")
del rt.transformer
torch.cuda.empty_cache()
vram("model freed")
bench_attn()