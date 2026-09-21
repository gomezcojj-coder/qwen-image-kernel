# Warm-path validation: consecutive edits with different prompts/images.
# gen1: cold (TE cycle + DiT load)   gen2/3: warm (slab + DiT resident)
import sys, time

import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True, ref_resolution=1024, verbose=True))

def gen(prompt, ref, seed, tag):
    marks = []
    t0 = time.perf_counter()
    img = rt.generate(prompt, image=[ref], output_resolution=1024, num_inference_steps=40, seed=seed,
                      step_callback=lambda i, t: marks.append(time.perf_counter()))[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    step_s = (marks[-1] - marks[0]) / max(len(marks) - 1, 1)
    print(f"{tag}: total {total:.1f}s | step0 {marks[0]-t0:.2f}s | steps {step_s:.3f}s", flush=True)
    return img

gen("Repaint the teapot with a glossy cobalt blue glaze, keep everything else identical",
    "bench_out/kernel.png", 42, "gen1 cold (teapot recolor)")
gen("Place this red sphere on a sandy beach at sunset, with its shadow, photorealistic",
    "bench_out/red_orb_rgba.png", 3, "gen2 warm (orb beach)")
gen("Make the teapot out of polished copper",
    "bench_out/kernel.png", 7, "gen3 warm (copper teapot)")