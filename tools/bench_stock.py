# Stock diffusers baseline: BF16 pipeline with sequential CPU offload (the
# only mode that fits 12 GB VRAM / 32 GB RAM). Same prompt/seed/steps/size.
import json
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from diffusers import DiffusionPipeline

PROMPT = "A ceramic teapot on a wooden table, morning light, photorealistic"
STEPS = 40
SEED = 42

print("loading stock pipeline (bf16, all components)...", flush=True)
t0 = time.perf_counter()
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
print(f"loaded+offload in {time.perf_counter() - t0:.1f}s", flush=True)

gen = torch.Generator("cuda").manual_seed(SEED)
print("stock warmup run...", flush=True)
t0 = time.perf_counter()
img = pipe(PROMPT, num_inference_steps=STEPS, height=1024, width=1024,
           generator=gen, output_type="pil").images[0]
print(f"stock warmup: {time.perf_counter() - t0:.2f}s", flush=True)

gen = torch.Generator("cuda").manual_seed(SEED)
marks = []
t0 = time.perf_counter()
img = pipe(PROMPT, num_inference_steps=STEPS, height=1024, width=1024,
           generator=gen, output_type="pil",
           callback_on_step_end=lambda p, i, t, kw: marks.append(time.perf_counter()) or kw).images[0]
torch.cuda.synchronize()
total = time.perf_counter() - t0
step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1) if len(marks) > 1 else 0.0
print(f"stock timed: {total:.2f}s total | {step_s:.3f}s/step", flush=True)

img.save("bench_out/stock.png")
with open("bench_out/stock_stats.json", "w") as f:
    json.dump({"total_s": round(total, 2), "steps": STEPS, "step_s": round(step_s, 4)}, f, indent=2)
print("saved bench_out/stock.png")