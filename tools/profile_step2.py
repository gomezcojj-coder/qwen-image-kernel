# Profile the current kernel runtime at 1024 t2i: per-step wall times
# (prefill vs decode) + a torch.profiler table over a few decode steps.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

os.chdir(Path(__file__).resolve().parent.parent)

import torch

from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

MODEL_ID = r"C:\Users\gomez\.cache\huggingface\hub\models--Qwen--Qwen-Image-2.1\snapshots\790c92633540aa0cb11d9abf19eb46d861714758"

OUT = Path("bench_out")
OUT.mkdir(exist_ok=True)
PROMPT = "A ceramic teapot on a wooden table, morning light, photorealistic"

rt = KernelRuntime(KernelRuntimeConfig(model_id=MODEL_ID))

step_times: list[float] = []
state = {"i": 0}


def cb(i, t):
    torch.cuda.synchronize()
    step_times.append(time.perf_counter())


t0 = time.perf_counter()
imgs = rt.generate(PROMPT, output_resolution=1024, num_inference_steps=40, seed=42,
                   step_callback=lambda i, t: cb(i, t))
total = time.perf_counter() - t0
print(f"[profile] total {total:.2f}s, final step sync {time.perf_counter()-step_times[-1]:.3f}s")
d = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
print(f"[profile] prefill step0->1 {d[0]:.3f}s; decode avg {sum(d[1:])/(len(d)-1):.3f}s; "
      f"decode min {min(d[1:]):.3f}s max {max(d[1:]):.3f}s")

# profile a few decode steps in a second generation (embed cache makes encode cheap)
from torch.profiler import ProfilerActivity, profile

times2: list[float] = []
state2 = {"n": 0}
prof = {"ctx": None}


def cb2(i, t):
    torch.cuda.synchronize()
    times2.append(time.perf_counter())
    n = state2["n"]
    state2["n"] += 1
    if n == 3:
        prof["ctx"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof["ctx"].__enter__()
    if n == 7 and prof["ctx"] is not None:
        prof["ctx"].__exit__(None, None, None)
        with open(OUT / "prof_decode.txt", "w") as f:
            print(prof["ctx"].key_averages().table(sort_by="self_cuda_time_total", row_limit=25), file=f)
        print("[profile] wrote bench_out/prof_decode.txt")


imgs = rt.generate(PROMPT, output_resolution=1024, num_inference_steps=40, seed=42, step_callback=cb2)
d2 = [times2[i + 1] - times2[i] for i in range(len(times2) - 1)]
print(f"[profile] run2 decode avg {sum(d2)/(len(d2)-1):.3f}s")
