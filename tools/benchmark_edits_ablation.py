# Completion run: the sections the first kernel benchmark missed.
#   edits: 768-ref x 2 prompts; 1024-ref accuracy run x 1
#   ablation: FP8-only (stock attention/elementwise) @1024 x 2 prompts
# Outputs: bench_out/kernel_bench2.json (merged into results later)
import json
import sys
import time

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

OUT = "bench_out/big"
STEPS = 40
SEED = 42
PROMPT = "A ceramic teapot on a wooden table, morning light, photorealistic"
PROMPT3 = "A misty mountain valley at dawn, layered ridges, long exposure, ultra detailed"
EDIT_PROMPTS = [
    ("Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical", "bench_out/kernel.png"),
    ("Place this red sphere on a sandy beach at sunset, with its shadow, photorealistic", "bench_out/red_orb_rgba.png"),
]

results = {"edit": [], "t2i_fp8only": []}

print("edits (config A)", flush=True)
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
for i, (eprompt, ref) in enumerate(EDIT_PROMPTS):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    t0 = time.perf_counter()
    img = rt.generate(eprompt, image=[ref], output_resolution=768, num_inference_steps=STEPS, seed=SEED,
                      step_callback=lambda j, t: marks.append(time.perf_counter()))[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    img.save(f"{OUT}/edit_768_e{i}.png")
    results["edit"].append({"cfg": "A", "ref_res": 768, "edit": i,
                            "total_s": round(total, 2),
                            "mean_step_s": round((marks[-1] - marks[0]) / max(len(marks) - 1, 1), 4),
                            "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    print(f"A edit 768 e{i}: {total:.2f}s", flush=True)

print("1024-ref edit (accuracy run)", flush=True)
del rt
torch.cuda.empty_cache()
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, ref_resolution=1024))
torch.cuda.reset_peak_memory_stats()
marks = []
t0 = time.perf_counter()
img = rt.generate(EDIT_PROMPTS[0][0], image=[EDIT_PROMPTS[0][1]], output_resolution=1024,
                  num_inference_steps=STEPS, seed=SEED,
                  step_callback=lambda j, t: marks.append(time.perf_counter()))[0]
torch.cuda.synchronize()
total = time.perf_counter() - t0
img.save(f"{OUT}/edit_1024_e0.png")
results["edit"].append({"cfg": "A", "ref_res": 1024, "edit": 0,
                        "total_s": round(total, 2),
                        "mean_step_s": round((marks[-1] - marks[0]) / max(len(marks) - 1, 1), 4),
                        "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
print(f"A edit 1024-ref: {total:.2f}s", flush=True)

print("CONFIG C: FP8 only (no fused kernels)", flush=True)
del rt
torch.cuda.empty_cache()
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=False))
img = rt.generate(PROMPT, output_resolution=1024, num_inference_steps=STEPS, seed=SEED)[0]  # warmup
for i, p in ((0, PROMPT), (3, PROMPT3)):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    t0 = time.perf_counter()
    img = rt.generate(p, output_resolution=1024, num_inference_steps=STEPS, seed=SEED,
                      step_callback=lambda j, t: marks.append(time.perf_counter()))[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    img.save(f"{OUT}/t2i_1024_p{i}_fp8only.png")
    results["t2i_fp8only"].append({"cfg": "C", "res": 1024, "prompt": i,
                                   "total_s": round(total, 2),
                                   "mean_step_s": round((marks[-1] - marks[0]) / max(len(marks) - 1, 1), 4),
                                   "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    print(f"C t2i 1024 p{i}: {total:.2f}s", flush=True)

json.dump(results, open("bench_out/kernel_bench2.json", "w"), indent=2)
print("saved bench_out/kernel_bench2.json")