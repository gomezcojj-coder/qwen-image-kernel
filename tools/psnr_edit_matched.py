# Matched-resolution edit PSNR: kernel fp8 vs stock bf16, both 768px output,
# same 768^2 reference resize, same seed.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

PROMPT = "Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical"
REF = "bench_out/kernel.png"
STEPS, SEED, RES = 40, 42, 768

# --- kernel fp8 ---
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
img = rt.generate(PROMPT, image=[REF], output_resolution=RES, num_inference_steps=STEPS,
                  seed=SEED)[0]
img.save("bench_out/edit768_kernel.png")
print("kernel 768 edit saved", flush=True)

# --- stock bf16 ---
from diffusers import DiffusionPipeline
from PIL import Image
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
gen = torch.Generator("cuda").manual_seed(SEED)
img2 = pipe(PROMPT, image=Image.open(REF), num_inference_steps=STEPS,
            output_resolution=RES, generator=gen, output_type="pil").images[0]
img2.save("bench_out/edit768_stock.png")
print("stock 768 edit saved", flush=True)

from qwen_image_kernel.metrics import image_psnr
p = image_psnr("bench_out/edit768_kernel.png", "bench_out/edit768_stock.png")
print(f"PSNR(kernel vs stock, matched 768): {p:.2f} dB")