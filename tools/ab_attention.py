# A/B: same kernel, same shapes - standalone vs inside the model forward.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.utils.torch_utils import randn_tensor
from qwen_image_kernel.triton_attention import block_causal_flash_attention

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
embeds, mask, img_pad = rt.encode_prompt("Repaint the teapot with a glossy cobalt blue glaze", image=["bench_out/kernel.png"])
rt.transformer = rt._build_transformer()

B, C, H, D = 1, 64, 32, 128
latents = randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
cond = randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
lmi = torch.cat([cond, latents], dim=1)
ipm = torch.cat([img_pad, img_pad.new_ones(B, 1024)], dim=1)
img_shapes = [[(1, 64, 64), (1, 64, 64)]]
cache = QwenImage21KVCache(32)
ts = torch.tensor([0.5], device=rt.device, dtype=torch.bfloat16)

def step(mode):
    return rt.transformer(
        hidden_states=lmi, timestep=ts,
        encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
        img_shapes=img_shapes, img_mask=ipm,
        attention_kwargs=None, kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]

step("extract")
step("cached")
torch.cuda.synchronize()

times = []
for _ in range(5):
    t0 = time.perf_counter()
    step("cached")
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
print("in-model cached step:", [f"{t:.2f}" for t in times], "s", flush=True)

# standalone, same shapes as the in-model call
P = 27 + 4096  # text(27) + cond(4096)
q = torch.randn(1, 4096, H, D, device=rt.device, dtype=torch.bfloat16) * 0.3
k_new = torch.randn(1, 4096, H, D, device=rt.device, dtype=torch.bfloat16) * 0.3
v_new = torch.randn_like(k_new)
ck = torch.randn(1, P, H, D, device=rt.device, dtype=torch.bfloat16) * 0.3
cv = torch.randn_like(ck)
L = torch.full((4096,), P + 4096 - 1, dtype=torch.int32, device=rt.device)
for _ in range(3):
    block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
torch.cuda.synchronize()
print(f"standalone kernel: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/call", flush=True)

# what does the wrapper actually choose / pass?
import inspect
from qwen_image_kernel import triton_attention as ta
import inspect
src = inspect.getsource(ta.block_causal_flash_attention)
print("wrapper config branch for SQ=4096:", "SQ <= 256" in src)