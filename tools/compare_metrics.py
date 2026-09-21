# Metrics + heatmap from the two saved comparison images.
import json

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

img_k = Image.open("bench_out/cmp_kernel.png").convert("RGB")
img_s = Image.open("bench_out/cmp_stock.png").convert("RGB")
a = np.asarray(img_k, dtype=np.float64) / 255.0
b = np.asarray(img_s, dtype=np.float64) / 255.0
diff = np.abs(a - b)

psnr = 10 * np.log10(1.0 / max((diff ** 2).mean(), 1e-12))
ssim = sk_ssim(a, b, channel_axis=2, data_range=1.0)

def dhash(im, hash_size=8):
    g = im.convert("L").resize((hash_size + 1, hash_size))
    d = np.asarray(g, dtype=np.int16)
    bits = d[:, :-1] > d[:, 1:]
    return bits.flatten()

hamming = int((dhash(img_k) != dhash(img_s)).sum())
frac_diff8 = (diff.max(axis=2) > 8 / 255).mean() * 100
frac_diff32 = (diff.max(axis=2) > 32 / 255).mean() * 100

dl = diff.mean(axis=2)
heat_img = np.stack([np.clip(dl * 6, 0, 1), np.clip(dl * 1.6, 0, 1), np.clip(dl * 1.2, 0, 1)], axis=2)
Image.fromarray((heat_img * 255).astype(np.uint8)).save("bench_out/cmp_diff_heat.png")

import hashlib
report = {
    "settings": {"prompt_sha1": hashlib.sha1(
        ("A polished modern anime illustration of a teenage girl standing at a rainy "
         "neon-lit train station at night, wearing a navy school uniform with a red "
         "ribbon, a transparent umbrella in one hand and a small black cat sitting "
         "beside her. Long dark-blue hair with subtle purple highlights, bright "
         "expressive eyes, soft blush, clean crisp cel-shading, detailed line art, "
         "dramatic rim lighting, reflections on wet pavement, glowing city lights in "
         "the background, cinematic composition, highly detailed background signage "
         "in Japanese style, and a station poster that clearly reads \"MIDNIGHT LINE\". "
         "The scene should feel emotional and atmospheric, with vivid colors, high "
         "contrast, and premium anime key visual quality.").encode()).hexdigest()[:12],
        "seed": 42, "res": 1024, "steps": 40, "cfg": 1.0},
    "kernel_fp8_total_s": 0, "stock_bf16_total_s": 290.9,
    "psnr_db": round(psnr, 2), "ssim": round(float(ssim), 4),
    "max_abs_diff_255": round(diff.max() * 255, 1),
    "mean_abs_diff_255": round(diff.mean() * 255, 3),
    "pct_pixels_diff_gt8": round(frac_diff8, 2),
    "pct_pixels_diff_gt32": round(frac_diff32, 2),
    "dhash_hamming_of_64": hamming,
}
json.dump(report, open("bench_out/cmp_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))