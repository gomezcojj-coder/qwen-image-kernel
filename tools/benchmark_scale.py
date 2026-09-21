# Large-scale kernel-runtime benchmark.
#
# Matrix:
#   t2i:  {512, 768, 1024} x 5 prompts  (config A = kernels + FP8)
#   ablation: fp8-only (no fused blocks) @1024 x 2 prompts
#   determinism: A @1024 prompt0 run twice -> PSNR
#   edits: 768-ref x 2 prompts; 1024-ref accuracy run x 1
# Every timed run is preceded by an untimed warmup at the same shape.
# Outputs: bench_out/big/*.png + kernel_bench.json
import json
import sys
import time

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from qwen_image_kernel.metrics import image_psnr

OUT = "bench_out/big"

PROMPTS = [
    "A ceramic teapot on a wooden table, morning light, photorealistic",
    'A cozy bookshop storefront with a hand-painted wooden sign that says "KERNEL" in gold letters, evening glow',
    "Portrait of an elderly fisherman with weathered skin, dramatic side lighting, 85mm lens",
    "A misty mountain valley at dawn, layered ridges, long exposure, ultra detailed",
    "An isometric cutaway of a tiny robot workshop, miniature style, soft lighting",
]
STEPS = 40
SEED = 42

EDIT_PROMPTS = [
    ("Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical", "bench_out/kernel.png"),
    ("Place this red sphere on a sandy beach at sunset, with its shadow, photorealistic", "bench_out/red_orb_rgba.png"),
]

results = {"t2i": [], "edit": [], "meta": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}}


def run_t2i(rt, res, prompt, seed=SEED, tag=""):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    t0 = time.perf_counter()
    rt.generate(prompt, output_resolution=res, num_inference_steps=STEPS, seed=seed,
                step_callback=lambda i, t: marks.append(time.perf_counter()))
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    return {"total_s": round(total, 2), "mean_step_s": round(step_s, 4), "vram_gib": round(vram, 2)}


def run_edit(rt, res, prompt, ref, tag=""):
    torch.cuda.reset_peak_memory_stats()
    marks = []
    t0 = time.perf_counter()
    rt.generate(prompt, image=[ref], output_resolution=res, num_inference_steps=STEPS, seed=SEED,
                step_callback=lambda i, t: marks.append(time.perf_counter()))
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    return {"total_s": round(total, 2), "mean_step_s": round(step_s, 4), "vram_gib": round(vram, 2)}


print("=" * 70)
print("CONFIG A: fused kernels + FP8 (the shipped runtime)", flush=True)
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
t_cold0 = time.perf_counter()
img = rt.generate(PROMPTS[0], output_resolution=1024, num_inference_steps=STEPS, seed=SEED)[0]
cold = time.perf_counter() - t_cold0
img.save(f"{OUT}/cold_start.png")
print(f"cold start (process->first image, incl. triton JIT): {cold:.1f}s", flush=True)
results["cold_start_s"] = round(cold, 2)

for res in (512, 768, 1024):
    for i, prompt in enumerate(PROMPTS):
        r = run_t2i(rt, res, prompt)
        img = rt.generate(PROMPTS[i], output_resolution=res, num_inference_steps=STEPS, seed=SEED)[0]
        img.save(f"{OUT}/t2i_{res}_p{i}.png")
        results["t2i"].append({"cfg": "A", "res": res, "prompt": i, **r})
        print(f"A t2i {res}px p{i}: {r['total_s']}s | {r['mean_step_s']}s/step | {r['vram_gib']} GiB", flush=True)

# determinism: rerun 1024/p0
img = rt.generate(PROMPTS[0], output_resolution=1024, num_inference_steps=STEPS, seed=SEED)[0]
img.save(f"{OUT}/t2i_1024_p0_rerun.png")
d = None
from PIL import Image as PILImage
import io, hashlib
a = hashlib.sha1(open(f"{OUT}/t2i_1024_p0.png", "rb").read()).hexdigest()
b = hashlib.sha1(open(f"{OUT}/t2i_1024_p0_rerun.png", "rb").read()).hexdigest()
results["determinism_bitexact"] = (a == b)
print(f"determinism (same seed, bytes identical): {a == b}", flush=True)

print("edits (config A)", flush=True)
for i, (eprompt, ref) in enumerate(EDIT_PROMPTS):
    r = run_edit(rt, 768, eprompt, ref)
    img = rt.generate(eprompt, image=[ref], output_resolution=768, num_inference_steps=STEPS, seed=SEED)[0]
    img.save(f"{OUT}/edit_768_e{i}.png")
    results["edit"].append({"cfg": "A", "ref_res": 768, "edit": i, **r})
    print(f"A edit 768 e{i}: {r['total_s']}s", flush=True)

del rt
torch.cuda.empty_cache()

print("=" * 70)
print("CONFIG A+ (1024 ref edit, accuracy run - auto-budget raises ref size)", flush=True)
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, ref_resolution=1024))
r = run_edit(rt, 1024, EDIT_PROMPTS[0][0], EDIT_PROMPTS[0][1])
img = rt.generate(EDIT_PROMPTS[0][0], image=[EDIT_PROMPTS[0][1]], output_resolution=1024,
                  num_inference_steps=STEPS, seed=SEED)[0]
img.save(f"{OUT}/edit_1024_e0.png")
results["edit"].append({"cfg": "A", "ref_res": 1024, "edit": 0, **r})
print(f"A edit 1024-ref: {r['total_s']}s", flush=True)
del rt
torch.cuda.empty_cache()

print("=" * 70)
print("CONFIG C: FP8 only (stock attention/elementwise, no fused kernels)", flush=True)
rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=False))
img = rt.generate(PROMPTS[0], output_resolution=1024, num_inference_steps=STEPS, seed=SEED)[0]  # warmup
for i in (0, 3):
    r = run_t2i(rt, 1024, PROMPTS[i])
    img = rt.generate(PROMPTS[i], output_resolution=1024, num_inference_steps=STEPS, seed=SEED)[0]
    img.save(f"{OUT}/t2i_1024_p{i}_fp8only.png")
    results["t2i"].append({"cfg": "C", "res": 1024, "prompt": i, **r})
    print(f"C t2i 1024 p{i}: {r['total_s']}s | {r['mean_step_s']}s/step", flush=True)

with open("bench_out/kernel_bench.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved bench_out/kernel_bench.json")