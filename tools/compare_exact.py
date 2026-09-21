# Exact-prompt, exact-seed comparison: kernel FP8 runtime vs stock BF16 pipeline.
# 1024x1024, 40 steps, CFG off, seed 42. Same prompt string byte-for-byte.
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")

PROMPT = (
    'A polished modern anime illustration of a teenage girl standing at a rainy '
    'neon-lit train station at night, wearing a navy school uniform with a red '
    'ribbon, a transparent umbrella in one hand and a small black cat sitting '
    'beside her. Long dark-blue hair with subtle purple highlights, bright '
    'expressive eyes, soft blush, clean crisp cel-shading, detailed line art, '
    'dramatic rim lighting, reflections on wet pavement, glowing city lights in '
    'the background, cinematic composition, highly detailed background signage '
    'in Japanese style, and a station poster that clearly reads "MIDNIGHT LINE". '
    'The scene should feel emotional and atmospheric, with vivid colors, high '
    'contrast, and premium anime key visual quality.'
)
SEED, RES, STEPS = 42, 1024, 40

print(f"prompt bytes: {len(PROMPT.encode('utf-8'))} | seed {SEED} | {RES}px | {STEPS} steps\n", flush=True)

# ---------------- kernel FP8 runtime ----------------
from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
t0 = time.perf_counter()
img_k = rt.generate(PROMPT, output_resolution=RES, num_inference_steps=STEPS, seed=SEED)[0]
tk = time.perf_counter() - t0
img_k.save("bench_out/cmp_kernel.png")
print(f"kernel FP8: {tk:.1f}s -> bench_out/cmp_kernel.png", flush=True)
del rt
torch.cuda.empty_cache()

# ---------------- stock BF16 (sequential offload) ----------------
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
gen = torch.Generator("cuda").manual_seed(SEED)
t0 = time.perf_counter()
img_s = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
             generator=gen, output_type="pil").images[0]
ts = time.perf_counter() - t0
img_s.save("bench_out/cmp_stock.png")
print(f"stock BF16: {ts:.1f}s -> bench_out/cmp_stock.png", flush=True)
del pipe
torch.cuda.empty_cache()

# ---------------- comparison ----------------
from skimage.metrics import structural_similarity as sk_ssim

a = np.asarray(img_k.convert("RGB"), dtype=np.float64) / 255.0
b = np.asarray(img_s.convert("RGB"), dtype=np.float64) / 255.0
diff = np.abs(a - b)

psnr = 10 * np.log10(1.0 / max((diff ** 2).mean(), 1e-12))
try:
    ssim = sk_ssim(a, b, channel_axis=2, data_range=1.0)
except TypeError:
    ssim = sk_ssim(a, b, multichannel=True, data_range=1.0)

# perceptual: dHash (9x8 grayscale -> 64-bit) Hamming distance
def dhash(im, hash_size=8):
    g = im.convert("L").resize((hash_size + 1, hash_size))
    d = np.asarray(g, dtype=np.int16)
    bits = d[:, :-1] > d[:, 1:]
    return bits.flatten()

bits_a, bits_b = dhash(img_k), dhash(img_s)
hamming = int((bits_a != bits_b).sum())

frac_diff8 = (diff.max(axis=2) > 8 / 255).mean() * 100
frac_diff32 = (diff.max(axis=2) > 32 / 255).mean() * 100

# diff heatmap (amplified x8, red overlay)
heat = np.clip(diff * 8 * 3, 0, 1)
heat_img = np.stack([np.clip(heat * 2.2, 0, 1), np.clip(diff * 2, 0, 1), np.clip(diff * 2, 0, 1)], axis=2)
from PIL import Image
Image.fromarray((heat_img * 255).astype(np.uint8)).save("bench_out/cmp_diff_heat.png")

report = {
    "settings": {"prompt_sha1": __import__("hashlib").sha1(PROMPT.encode()).hexdigest()[:12],
                 "seed": SEED, "res": RES, "steps": STEPS, "cfg": 1.0},
    "kernel_fp8": {"total_s": round(tk, 1)},
    "stock_bf16": {"total_s": round(ts, 1)},
    "psnr_db": round(10 * np.log10(1.0 / max((diff ** 2).mean(), 1e-12)), 2),
    "ssim": round(float(ssim), 4),
    "max_abs_diff_255": round(diff.max() * 255, 1),
    "mean_abs_diff_255": round(diff.mean() * 255, 3),
    "pct_pixels_diff_gt8": round(frac_diff8, 2),
    "pct_pixels_diff_gt32": round(frac_diff32, 2),
    "dhash_hamming_of_64": hamming,
}
import json
json.dump(report, open("bench_out/cmp_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))