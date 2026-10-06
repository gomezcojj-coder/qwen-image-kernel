# Timed benchmark of the optimized runtime (config A) using the RESULTS.md
# methodology: per shape, one untimed warmup then timed runs over 5 prompts.
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

os.chdir(Path(__file__).resolve().parent.parent)

import torch

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

MODEL_ID = r"C:\Users\gomez\.cache\huggingface\hub\models--Qwen--Qwen-Image-2.1\snapshots\790c92633540aa0cb11d9abf19eb46d861714758"
PROMPTS = [
    "A ceramic teapot on a wooden table, morning light, photorealistic",
    'A cozy bookshop storefront with a hand-painted wooden sign that says "KERNEL" in gold letters, evening glow',
    "Portrait of an elderly fisherman with weathered skin, dramatic side lighting, 85mm lens",
    "A misty mountain valley at dawn, layered ridges, long exposure, ultra detailed",
    "An isometric cutaway of a tiny robot workshop, miniature style, soft lighting",
]
OUT = Path("bench_out/fused_bench.json")

rt = KernelRuntime(KernelRuntimeConfig(model_id=MODEL_ID))
results = {"meta": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}, "t2i": []}
for res in (1024, 768, 512):
    # warmup at this shape (untimed)
    rt.generate(PROMPTS[0], output_resolution=res, num_inference_steps=40, seed=42)
    torch.cuda.reset_peak_memory_stats()
    for pi, prompt in enumerate(PROMPTS):
        times = []
        t0 = time.perf_counter()
        img = rt.generate(prompt, output_resolution=res, num_inference_steps=40, seed=42,
                          step_callback=lambda i, t: times.append(time.perf_counter()))
        total = time.perf_counter() - t0
        d = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        rec = {"cfg": "A-fused", "res": res, "prompt": pi, "total_s": round(total, 2),
               "mean_step_s": round(sum(d) / (len(d) - 1), 4),
               "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        results["t2i"].append(rec)
        print(f"[bench] {res}px p{pi}: {total:.1f}s total, {rec['mean_step_s']} s/step, "
              f"{rec['vram_gib']} GiB", flush=True)
        img[0].save(f"bench_out/fused_{res}_p{pi}.png")

OUT.write_text(json.dumps(results, indent=2))
print(f"[bench] wrote {OUT}")
