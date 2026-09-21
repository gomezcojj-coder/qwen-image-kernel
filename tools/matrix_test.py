# Matrix: does the PROMPT (text length) change the in-model step speed?
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.utils.torch_utils import randn_tensor
from qwen_image_kernel.triton_attention import block_causal_flash_attention

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
rt.transformer = rt._build_transformer()
B, C, H, D = 1, 64, 32, 128

def make_step(prompt):
    embeds, mask, img_pad = rt.encode_prompt(prompt, image=["bench_out/kernel.png"])
    text_n = embeds.shape[1] - int(img_pad.sum())
    target_slots = 1024
    lmi = torch.cat([
        randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2),
        randn_tensor((B, 1, C, 64, 64), device=rt.device, dtype=torch.bfloat16).view(B, C, 4096).transpose(1, 2),
    ], dim=1)
    ipm = torch.cat([img_pad, img_pad.new_ones(B, target_slots)], dim=1)
    img_shapes = [[(1, 64, 64), (1, 64, 64)]]
    cache = QwenImage21KVCache(32)
    ts = torch.tensor([0.5], device=rt.device, dtype=torch.bfloat16)

    def step(mode):
        return rt.transformer(
            hidden_states=lmi, timestep=ts,
            encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
            img_shapes=img_shapes, img_mask=ipm,
            attention_kwargs=None, kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]

    return step, text_n, embeds.shape[1]

for prompt in ["Repaint the teapot with a glossy cobalt blue glaze", "test"]:
    step, text_n, S = make_step(prompt)
    step("extract")
    times = []
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        step("cached")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"prompt={prompt[:20]!r:24s} text={text_n} S={S} steps: {[f'{t:.2f}' for t in times]}", flush=True)
    del cache