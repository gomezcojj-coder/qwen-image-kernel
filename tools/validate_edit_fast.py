# Validate: 1024-ref edit speed with the contiguous KV cache + fp8 V.
import sys, time

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

PROMPT = "Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical"
REF = "bench_out/kernel.png"

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, ref_resolution=1024, verbose=True))
for label, seed in (("run1", 42), ("run2", 42)):
    marks = []
    t0 = time.perf_counter()
    img = rt.generate(PROMPT, image=[REF], output_resolution=1024, num_inference_steps=40, seed=seed,
                      step_callback=lambda i, t: marks.append(time.perf_counter()))[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    prefill = marks[0] - t0
    img.save(f"bench_out/big/edit_1024_fast_{label}.png")
    print(f"{label}: total {total:.1f}s | step0(prefill) {prefill:.2f}s | steps {step_s:.3f}s", flush=True)

from qwen_image_kernel.metrics import image_psnr
p = image_psnr("bench_out/big/edit_1024_fast_run1.png", "bench_out/big/stock_edit_1024.png")
print(f"PSNR vs stock 1024-ref edit: {p:.2f} dB")