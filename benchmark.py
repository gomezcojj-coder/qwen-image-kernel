#!/usr/bin/env python
"""Benchmark: stock diffusers baseline vs kernel runtime, with PSNR accuracy.

  python benchmark.py --steps 40 --size 1024 --seed 42

Stages:
  1. kernel runtime (FP8 DiT + fused Triton kernels + prefix KV cache)
     -> image + wall time (+ step times)
  2. stock QwenImage21Pipeline (BF16, sequential CPU offload - the only mode
     that fits 12 GB) -> image + wall time
  3. PSNR(kernel, stock) + speedup table
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def bench_kernel(args, prompt: str, seed: int) -> tuple[dict, object]:
    from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

    cfg = KernelRuntimeConfig(fp8=True, kernel_blocks=True)
    rt = KernelRuntime(cfg)

    # warmup: triton JIT, cache allocations, cublas heuristics - not timed
    rt.generate_text_to_image(prompt, height=args.size, width=args.size,
                              num_inference_steps=args.steps, seed=seed)
    torch.cuda.synchronize()

    step_marks = []
    t0 = time.perf_counter()
    img = rt.generate_text_to_image(
        prompt, height=args.size, width=args.size,
        num_inference_steps=args.steps, seed=seed,
        step_callback=lambda i, t: step_marks.append(time.perf_counter()),
    )[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (step_marks[-1] - step_marks[0]) / max(len(step_marks) - 1, 1) if len(step_marks) > 1 else 0.0
    stats = {"total_s": round(total, 3), "steps": args.steps, "mean_step_s": round(step_s, 4)}
    return stats, img


def bench_stock(args, prompt: str, seed: int) -> tuple[dict, object]:
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.model, dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()  # the only mode that fits 12 GB VRAM

    gen = torch.Generator("cuda").manual_seed(seed)
    pipe(prompt, num_inference_steps=args.steps, height=args.size, width=args.size,
         generator=gen, output_type="pil")  # warmup
    torch.cuda.synchronize()

    gen = torch.Generator("cuda").manual_seed(seed)
    t0 = time.perf_counter()
    out = pipe(prompt, num_inference_steps=args.steps, height=args.size, width=args.size,
               generator=gen, output_type="pil")
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return {"total_s": round(total, 3)}, out.images[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A ceramic teapot on a wooden table, morning light, photorealistic")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--skip-stock", action="store_true", help="kernel stage only")
    ap.add_argument("--outdir", default="bench_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    kernel_img_path = os.path.join(args.outdir, "kernel.png")
    stock_img_path = os.path.join(args.outdir, "stock.png")
    results: dict = {}

    print("=" * 64)
    print("stage 1: kernel runtime (FP8 + fused Triton kernels + prefix KV cache)")
    k, kimg = bench_kernel(args, args.prompt, args.seed)
    kimg.save(kernel_img_path)
    results["kernel"] = k
    print(f"kernel : {k['total_s']:.2f}s total | {k['mean_step_s']:.3f}s/step "
          f"({args.steps} steps @ {args.size}px)")

    if not args.skip_stock:
        print("stage 2: stock diffusers pipeline (BF16, sequential offload)")
        s, simg = bench_stock(args, args.prompt, args.seed)
        simg.save(stock_img_path)
        results["stock"] = s
        from qwen_image_kernel.metrics import image_psnr
        p = image_psnr(kernel_img_path, stock_img_path)
        results["psnr_db"] = round(p, 2)
        print(f"stock  : {s['total_s']:.2f}s")
        print(f"PSNR(kernel vs stock BF16): {p:.2f} dB")
        print(f"speedup: {s['total_s'] / k['total_s']:.2f}x")

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())