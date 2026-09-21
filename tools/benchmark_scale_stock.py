# Large-scale stock baseline (BF16 + sequential CPU offload).
# t2i: {512, 768, 1024} x 2 prompts;  edits: 768-ref and 1024-ref recolor.
# Same seeds/prompts as the kernel benchmark for PSNR pairing.
import json
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from diffusers import DiffusionPipeline

OUT = "bench_out/big"

PROMPTS = [
    "A ceramic teapot on a wooden table, morning light, photorealistic",
    None,  # p1 unused (time bound) - index kept for pairing clarity
    None,
    "A misty mountain valley at dawn, layered ridges, long exposure, ultra detailed",
    None,
]
STOCK_PROMPTS = {0: PROMPTS[0], 3: PROMPTS[3]}
STEPS = 40
SEED = 42
EDIT_PROMPT = "Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical"
REF = "bench_out/kernel.png"

results = {"t2i": [], "edit": [], "meta": {"gpu": torch.cuda.get_device_name(0)}}

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

def run_t2i(res, prompt, tag):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    gen = torch.Generator("cuda").manual_seed(42)
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=STEPS, height=res, width=res,
               generator=gen, output_type="pil",
               callback_on_step_end=lambda p, i, t, kw: marks.append(time.perf_counter()) or kw).images[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    return {"total_s": round(total, 2), "mean_step_s": round(step_s, 4), "vram_gib": round(vram, 2)}, img

# warmup at the cheapest res
run_t2i(512, PROMPTS[0], "warm")
for res in (512, 768, 1024):
    for i, prompt in STOCK_PROMPTS.items():
        r, img = run_t2i(res, prompt, f"p{i}")
        img.save(f"{OUT}/stock_t2i_{res}_p{i}.png")
        results["t2i"].append({"cfg": "B", "res": res, "prompt": i, **r})
        print(f"B t2i {res}px p{i}: {r['total_s']}s", flush=True)

# edits (stock resizes refs to output_resolution^2 -> matched settings at 768;
# the 1024 run uses a 1024^2 reference like the kernel's accuracy run)
for res in (768, 1024):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    gen = torch.Generator("cuda").manual_seed(42)
    t0 = time.perf_counter()
    img = pipe(EDIT_PROMPT, image=__import__("PIL.Image", fromlist=["Image"]).open(REF),
               num_inference_steps=STEPS, output_resolution=res,
               generator=gen, output_type="pil",
               callback_on_step_end=lambda p, i, t, kw: marks.append(time.perf_counter()) or kw).images[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    img.save(f"{OUT}/stock_edit_{res}.png")
    results["edit"].append({"cfg": "B", "ref_res": res, "edit": 0,
                            "total_s": round(total, 2),
                            "mean_step_s": round((marks[-1] - marks[0]) / max(len(marks) - 1, 1), 4),
                            "vram_gib": round(vram, 2)})
    print(f"B edit {res}: {total:.1f}s", flush=True)

with open("bench_out/stock_bench.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved bench_out/stock_bench.json")