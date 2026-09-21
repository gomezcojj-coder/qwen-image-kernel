# Profile the edit path: phase timings + step times + memory.
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig
from PIL import Image

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, verbose=True))

marks = []
class T:
    pass
t = T()
orig = rt.generate
import types

# monkeypatch the scheduler loop via callback only - time the phases from outside
PROMPT = "Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical"
REF = "bench_out/kernel.png"

t0 = time.perf_counter()
def cb(i, tt):
    if i in (0, 1, 2, 10, 20, 39):
        marks.append((i, time.perf_counter() - t0))

img = rt.generate(
    PROMPT, image=[REF], num_inference_steps=40, seed=42,
    output_resolution=1024, step_callback=cb,
)[0]
total = time.perf_counter() - t0
img.save("bench_out/edit_blue_prof.png")
print(f"TOTAL: {total:.2f}s")
for i, dt in marks:
    print(f"  step {i:2d} done at {dt:7.2f}s")
print("last step duration:", round(marks[-1][1] - marks[-2][1], 3), "s (from step marks)")