# Verify the fused/optimized path's accuracy vs the stock BF16 references at
# the same configs the RESULTS.md table used (seed 42, 40 steps).
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

os.chdir(Path(__file__).resolve().parent.parent)

import torch

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from qwen_image_kernel.metrics import image_psnr

MODEL_ID = r"C:\Users\gomez\.cache\huggingface\hub\models--Qwen--Qwen-Image-2.1\snapshots\790c92633540aa0cb11d9abf19eb46d861714758"

PROMPTS = [
    "A ceramic teapot on a wooden table, morning light, photorealistic",
    'A cozy bookshop storefront with a hand-painted wooden sign that says "KERNEL" in gold letters, evening glow',
    "Portrait of an elderly fisherman with weathered skin, dramatic side lighting, 85mm lens",
    "A misty mountain valley at dawn, layered ridges, long exposure, ultra detailed",
    "An isometric cutaway of a tiny robot workshop, miniature style, soft lighting",
]
RUNS = [(512, 0), (512, 3), (768, 0), (768, 3), (1024, 0), (1024, 3)]
PREV = {(512, 0): 28.15, (512, 3): 29.94, (768, 0): 22.22, (768, 3): 36.50,
        (1024, 0): 29.83, (1024, 3): 27.49}

rt = KernelRuntime(KernelRuntimeConfig(model_id=MODEL_ID))
for res, pi in RUNS:
    times = []
    img = rt.generate(PROMPTS[pi], output_resolution=res, num_inference_steps=40, seed=42,
                      step_callback=lambda i, t: times.append(time.perf_counter()))
    out = Path(f"bench_out/fused_t2i_{res}_p{pi}.png")
    img[0].save(out)
    d = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    ref = Path(f"bench_out/big/stock_t2i_{res}_p{pi}.png")
    psnr = image_psnr(str(out), str(ref))
    print(f"[verify] {res}px p{pi}: PSNR {psnr:.2f} dB (prev {PREV[(res, pi)]:.2f} dB), "
          f"decode {sum(d)/(len(d)-1):.3f} s/step", flush=True)
