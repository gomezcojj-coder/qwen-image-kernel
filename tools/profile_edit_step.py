# Micro-profile an EDIT-mode decode step: where do the 8.8 s go?
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from qwen_image_kernel.runtime import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
import numpy as np

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
embeds, mask, img_pad = rt.encode_prompt("Repaint the teapot with a glossy cobalt blue glaze", image=["bench_out/kernel.png"])
print("embeds:", tuple(embeds.shape), "img slots:", int(img_pad.sum()), flush=True)

rt.transformer = rt._build_transformer()

B, C = 1, 64
lat_h = lat_w = 64
latents = randn_tensor((B, 1, C, lat_h, lat_w), device=rt.device, dtype=torch.bfloat16)
latents = latents.view(B, C, lat_h * lat_w).transpose(1, 2)
seq_len = latents.shape[1]
# cond latent tokens: 1024x1024 ref -> 64x64 = 4096
cond_latents = randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16)
cond_latents = cond_latents.view(B, C, 4096).transpose(1, 2)
latent_model_input = torch.cat([cond_latents, latents], dim=1)

target_slots = seq_len // 4
ipm = torch.cat([img_pad, img_pad.new_ones(B, target_slots)], dim=1)
img_shapes = [[(1, 64, 64), (1, 64, 64)]]

cache = QwenImage21KVCache(32)
ipm = torch.cat([img_pad, img_pad.new_ones(B, target_slots)], dim=1)
t0 = time.perf_counter()
noise = rt.transformer(
    hidden_states=latent_model_input, timestep=torch.tensor([0.9], device=rt.device, dtype=torch.bfloat16),
    encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
    img_shapes=img_shapes, img_mask=ipm,
    attention_kwargs=None, kv_cache=cache, kv_cache_mode="extract", return_dict=False)[0]
torch.cuda.synchronize()
print(f"extract (prefill 8697 tokens): {time.perf_counter() - t0:.2f}s", flush=True)

# warm one cached step
t0 = time.perf_counter()
noise = rt.transformer(
    hidden_states=latent_model_input, timestep=torch.tensor([0.5], device=rt.device, dtype=torch.bfloat16),
    encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
    img_shapes=img_shapes, img_mask=ipm,
    attention_kwargs=None, kv_cache=cache, kv_cache_mode="cached", return_dict=False)[0]
torch.cuda.synchronize()
print(f"cached step (first): {time.perf_counter() - t0:.2f}s", flush=True)

for _ in range(3):
    t0 = time.perf_counter()
    noise = rt.transformer(
        hidden_states=latent_model_input, timestep=torch.tensor([0.5], device=rt.device, dtype=torch.bfloat16),
        encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
        img_shapes=img_shapes, img_mask=ipm,
        attention_kwargs=None, kv_cache=cache, kv_cache_mode="cached", return_dict=False)[0]
    torch.cuda.synchronize()
    print(f"cached step: {time.perf_counter() - t0:.2f}s", flush=True)

# time attention kernel alone at edit decode shapes
from qwen_image_kernel.processors import _build_prefix_lengths  # noqa
from qwen_image_kernel.triton_attention import block_causal_flash_attention
q = torch.randn(1, 4096, 32, 128, device=rt.device, dtype=torch.bfloat16)
k_new = torch.randn(1, 4096, 32, 128, device=rt.device, dtype=torch.bfloat16)
v_new = torch.randn_like(k_new)
ck = torch.randn(1, 4537, 32, 128, device=rt.device, dtype=torch.bfloat16)
cv = torch.randn_like(ck)
L = torch.full((4096,), 8632, dtype=torch.int32, device=rt.device)
for _ in range(2):
    out = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
torch.cuda.synchronize()
print(f"attention kernel alone (4096q, 8633kv): {(time.perf_counter() - t0) / 10 * 1000:.1f} ms/call", flush=True)