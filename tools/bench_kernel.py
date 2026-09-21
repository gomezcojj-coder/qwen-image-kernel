# Timed kernel-runtime benchmark: 1 warmup + N timed runs.
import json
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

PROMPT = "A ceramic teapot on a wooden table, morning light, photorealistic"
STEPS = 40
SEED = 42

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, verbose=True))

print("warmup run (triton JIT, cache warm)...", flush=True)
t0 = time.perf_counter()
img = rt.generate_text_to_image(PROMPT, 1024, 1024, STEPS, seed=SEED)[0]
print(f"warmup (includes triton compile): {time.perf_counter() - t0:.2f}s", flush=True)

# timed run 1: cold caches (embed cache on), full pipeline
marks = []
t0 = time.perf_counter()
img = rt.generate_text_to_image(PROMPT, 1024, 1024, STEPS, seed=SEED,
                                step_callback=lambda i, t: marks.append(time.perf_counter()))[0]
torch.cuda.synchronize()
run1 = time.perf_counter() - t0
step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
print(f"timed run 1: {run1:.2f}s total | {step_s:.3f}s/step (denoise)", flush=True)

# timed run 2: stability check
marks2 = []
t0 = time.perf_counter()
img2 = rt.generate_text_to_image(PROMPT, 1024, 1024, STEPS, seed=SEED,
                                 step_callback=lambda i, t: marks2.append(time.perf_counter()))[0]
torch.cuda.synchronize()
run2 = time.perf_counter() - t0
step_s2 = (marks2[-1] - marks2[0]) / max(len(marks2) - 1, 1)
print(f"timed run 2: {run2:.2f}s total | {step_s2:.3f}s/step (denoise)", flush=True)

img.save("bench_out/kernel.png")
with open("bench_out/kernel_stats.json", "w") as f:
    json.dump({"run1_s": round(run1, 2), "run2_s": round(run2, 2),
               "step_s": round(step_s, 4), "steps": STEPS}, f, indent=2)
print("saved bench_out/kernel.png")