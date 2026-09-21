# Generate the missing stock BF16 sides for the 1024px pairs (prompts 1, 2, 4).
import sys
import time

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)
from diffusers import DiffusionPipeline

PROMPTS = {
    1: 'A cozy bookshop storefront with a hand-painted wooden sign that says "KERNEL" in gold letters, evening glow',
    2: "Portrait of an elderly fisherman with weathered skin, dramatic side lighting, 85mm lens",
    4: "An isometric cutaway of a tiny robot workshop, miniature style, soft lighting",
}
STEPS, SEED, RES = 40, 42, 1024

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

for i, prompt in PROMPTS.items():
    gen = torch.Generator("cuda").manual_seed(SEED)
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=STEPS, height=RES, width=RES,
               generator=gen, output_type="pil").images[0]
    dt = time.perf_counter() - t0
    img.save(f"bench_out/big/stock_t2i_1024_p{i}.png")
    print(f"stock 1024 p{i}: {dt:.1f}s", flush=True)