# Profile ONE edit-mode cached decode step with torch.profiler.
import sys

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.utils.torch_utils import randn_tensor

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
embeds, mask, img_pad = rt.encode_prompt("Repaint the teapot with a glossy cobalt blue glaze", image=["bench_out/kernel.png"])
rt.transformer = rt._build_transformer()

B, C = 1, 64
latents = randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
cond = randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2)
lmi = torch.cat([cond, latents], dim=1)
ipm = torch.cat([img_pad, img_pad.new_ones(B, 1024)], dim=1)
img_shapes = [[(1, 64, 64), (1, 64, 64)]]
cache = QwenImage21KVCache(32)
ts = torch.tensor([0.5], device=rt.device, dtype=torch.bfloat16)

def step(mode="cached"):
    return rt.transformer(
        hidden_states=lmi, timestep=ts,
        encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
        img_shapes=img_shapes, img_mask=ipm,
        attention_kwargs=None, kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]

step("extract")  # populate the prefix cache
step()  # warm
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))