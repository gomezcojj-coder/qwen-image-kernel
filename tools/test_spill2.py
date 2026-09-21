# Pin down what poisons the kernel: model only vs model+steps(cache resident).
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
    print(f"{tag:36s} free {free / 2**30:5.2f} GiB", flush=True)

def bench_attn(tag):
    ms = bench(lambda: block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv))
    print(f"{tag:36s} {ms:8.2f} ms/call")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.utils.torch_utils import randn_tensor

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
embeds, mask, img_pad = rt.encode_prompt("test", image=["bench_out/kernel.png"])
rt.transformer = rt._build_transformer()

B, C = 1, 64
latents = randn_tensor((B, 1, C, 64, 64), device=DEV, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
cond = randn_tensor((B, 1, C, 64, 64), device=DEV, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
lmi = torch.cat([cond, latents], dim=1)
ipm = torch.cat([img_pad, img_pad.new_ones(B, 1024)], dim=1)
img_shapes = [[(1, 64, 64), (1, 64, 64)]]
cache = QwenImage21KVCache(32)
ts = torch.tensor([0.5], device=DEV, dtype=torch.bfloat16)

def step(mode):
    return rt.transformer(
        hidden_states=lmi, timestep=ts,
        encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
        img_shapes=img_shapes, img_mask=ipm,
        attention_kwargs=None, kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]

vram("model loaded, before steps")
bench_attn("bench (model, no cache)")

step("extract")
vram("model + cache resident")
bench_attn("bench (model + steps done)")

step("cached")
step("cached")
vram("after 2 cached steps")
bench_attn("bench (model + steps done)")

# free just the cache, keep model
del cache
torch.cuda.empty_cache()
vram("cache freed")
bench_attn("bench (cache freed)")