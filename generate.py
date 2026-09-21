#!/usr/bin/env python
"""Generate images with Qwen-Image-2.1 on the kernel runtime.

Usage:
  python generate.py --prompt "a ceramic teapot on a wooden table" --steps 40 --size 1024
  python generate.py --prompt "..." --seed 7 --out my.png --fp8 --kernel
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = __import__("argparse").ArgumentParser(description="Qwen-Image-2.1 kernel runtime")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", nargs="*", default=None, help="condition image path(s) for editing")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--true-cfg", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=1024, help="max side / output resolution (1024/768/512)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="output.png")
    ap.add_argument("--fp8", action="store_true", default=True, help="FP8 weights (default on)")
    ap.add_argument("--no-fp8", dest="fp8", action="store_false")
    ap.add_argument("--no-kernel-blocks", dest="kernel", action="store_false", help="fused kernels off (attention only)")
    ap.add_argument("--warmup", type=int, default=0, help="extra warm generations before timing")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

    cfg = KernelRuntimeConfig(fp8=args.fp8, kernel_blocks=args.kernel)
    rt = KernelRuntime(cfg)

    t0 = time.perf_counter()
    images = rt.generate(
        args.prompt,
        image=args.image,
        output_resolution=args.size,
        num_inference_steps=args.steps,
        seed=args.seed,
        true_cfg_scale=args.true_cfg,
        negative_prompt=args.negative,
    )
    dt = time.perf_counter() - t0
    images[0].save(args.out)
    print(f"[generate] saved {args.out} in {dt:.2f}s ({args.steps} steps, {args.size}px, fp8={args.fp8}, kernels={args.kernel})")
    return 0


if __name__ == "__main__":
    sys.exit(main())