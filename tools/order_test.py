# Order test: encode-then-build vs build-then-encode.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.utils.torch_utils import randn_tensor

PROMPT = "Repaint the teapot with a glossy cobalt blue glaze"

def run(order):
    rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
    if order == "encode_first":
        embeds, mask, img_pad = rt.encode_prompt(PROMPT, image=["bench_out/kernel.png"])
        rt.transformer = rt._build_transformer()
    else:
        rt.transformer = rt._build_transformer()
        embeds, mask, img_pad = rt.encode_prompt(PROMPT, image=["bench_out/kernel.png"])

    B, C = 1, 64
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
    times = []
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        step("cached")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"order={order:12s} cached steps: {[f'{t:.2f}' for t in times]}s", flush=True)

run("encode_first")
run("build_first")