# Compute PSNR/SSIM for all matched pairs -> bench_out/pair_metrics.json
import json
import sys

sys.path.insert(0, r"C:\Users\gomez\Desktop\Coding_Projects\Qwen_Image")
from skimage.metrics import structural_similarity as sk_ssim
from PIL import Image
import numpy as np


def metrics(a_path, b_path):
    x = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float64) / 255
    y = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float64) / 255
    d = np.abs(x - y)
    psnr = 10 * np.log10(1.0 / max((d ** 2).mean(), 1e-12))
    ssim = sk_ssim(x, y, channel_axis=2, data_range=1.0)
    return round(psnr, 2), round(float(ssim), 4)


pairs = [
    ("anime girl @1024", "bench_out/cmp_kernel.png", "bench_out/cmp_stock.png"),
    ("text sign @1024", "bench_out/big/t2i_1024_p1.png", "bench_out/big/stock_t2i_1024_p1.png"),
    ("portrait @1024", "bench_out/big/t2i_1024_p2.png", "bench_out/big/stock_t2i_1024_p2.png"),
    ("valley @1024", "bench_out/big/t2i_1024_p3.png", "bench_out/big/stock_t2i_1024_p3.png"),
    ("workshop @1024", "bench_out/big/t2i_1024_p4.png", "bench_out/big/stock_t2i_1024_p4.png"),
    ("teapot @512", "bench_out/big/t2i_512_p0.png", "bench_out/big/stock_t2i_512_p0.png"),
    ("edit recolor @768", "bench_out/big/edit_768_e0.png", "bench_out/big/stock_edit_768.png"),
]
out = {}
for name, k, s in pairs:
    p, ss = metrics(k, s)
    out[name] = {"psnr_db": p, "ssim": ss}
    print(f"{name:20s} PSNR {p:6.2f} dB  SSIM {ss:.4f}")
json.dump(out, open("bench_out/pair_metrics.json", "w"), indent=2)