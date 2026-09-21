# Stock BF16 edit baseline (sequential offload, fp32 VAE), same prompt/image/seed.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from diffusers import DiffusionPipeline
from PIL import Image

PROMPT = "Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical"
REF = "bench_out/kernel.png"
STEPS, SEED = 40, 42

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

ref = Image.open(REF)
gen = torch.Generator("cuda").manual_seed(SEED)
t0 = time.perf_counter()
img = pipe(PROMPT, image=ref, num_inference_steps=STEPS, output_resolution=1024,
           generator=gen, output_type="pil").images[0]
total = time.perf_counter() - t0
img.save("bench_out/stock_edit.png")
print(f"stock edit: {total:.1f}s -> bench_out/stock_edit.png")