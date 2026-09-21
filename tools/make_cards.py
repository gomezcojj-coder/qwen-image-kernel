# Build labeled side-by-side comparison cards (kernel FP8 vs stock BF16) with
# per-pair PSNR/SSIM captions and optional zoom crops for detail regions.
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as sk_ssim

BIG = "bench_out/big"


def metrics(a: Image.Image, b: Image.Image):
    x = np.asarray(a.convert("RGB"), dtype=np.float64) / 255
    y = np.asarray(b.convert("RGB"), dtype=np.float64) / 255
    d = np.abs(x - y)
    psnr = 10 * np.log10(1.0 / max((d ** 2).mean(), 1e-12))
    try:
        ssim = sk_ssim(x, y, channel_axis=2, data_range=1.0)
    except TypeError:
        ssim = sk_ssim(x, y, multichannel=True, data_range=1.0)
    return psnr, ssim


def font(size=22):
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def card(kernel_path, stock_path, out, caption_l, caption_r, metric_line, crops=None):
    """Side-by-side card: 768px halves + 26px caption bars; optional zoom-crop rows."""
    k = Image.open(kernel_path).convert("RGB")
    s = Image.open(stock_path).convert("RGB")
    half = 768
    kh = k.resize((half, int(k.height * half / k.width)), Image.LANCZOS)
    sh = s.resize((half, int(s.height * half / s.width)), Image.LANCZOS)
    bar, gap = 34, 10
    rows = [max(kh.height, sh.height)]
    crop_imgs = []
    if crops:
        for (box, label) in crops:
            kc = k.crop(box)
            scale = 560 / kc.width
            kc = kc.resize((560, int(kc.height * scale)), Image.LANCZOS)
            sc = s.crop(box).resize(kc.size, Image.LANCZOS)
            rows.append(max(kc.height, sc.height) + bar)
            crop_imgs.append((kc, sc, label))

    W = half * 2 + gap
    H = bar + sum(rows) + gap * (len(rows))
    canvas = Image.new("RGB", (W, H), (10, 12, 17))
    dr = ImageDraw.Draw(canvas)
    f = font(21); fs = font(19)
    y = 0
    dr.rectangle([0, y, W, y + bar], fill=(24, 28, 38))
    dr.text((12, y + 6), f"kernel FP8 · {caption_l}", font=f, fill=(140, 180, 255))
    dr.text((half + gap + 8, y + 6), f"stock BF16 · {caption_r}", font=f, fill=(230, 200, 140))
    y += bar
    canvas.paste(kh, (0, y)); canvas.paste(sh, (half + gap, y))
    y += rows[0] + gap
    if metric_line:
        dr.rectangle([0, y - 2, W, y + 26], fill=(14, 17, 24))
        dr.text((10, y), metric_line, font=fs, fill=(160, 170, 190))
        y += 30
    for (kc, sc, label) in crop_imgs:
        dr.rectangle([0, y, W, y + bar], fill=(14, 17, 24))
        dr.text((8, y + 6), label, font=fs, fill=(200, 200, 210))
        y += bar
        canvas.paste(kc, (0, y)); canvas.paste(sc, (half + gap, y))
        y += max(kc.height, sc.height) + gap
    canvas.save(out, quality=92)
    print("saved", out)


def m(kernel_path, stock_path):
    a = Image.open(kernel_path).convert("RGB")
    b = Image.open(stock_path).convert("RGB")
    p, s = metrics(a, b)
    return f"PSNR {p:.1f} dB · SSIM {s:.3f} · same prompt + seed 42"


# 1) anime girl pair (the flagship)
card("bench_out/cmp_kernel.png", "bench_out/cmp_stock.png", "docs/img/pair_anime.png",
     "74.7 s (cold) · 7.6 GiB", "290.9 s", m("bench_out/cmp_kernel.png", "bench_out/cmp_stock.png"),
     crops=[((300, 250, 868, 810), "zoom: face / line art"),
            ((40, 520, 608, 1024), "zoom: signage / pavement reflections")])

# 2) text rendering — the KERNEL sign
card("bench_out/big/t2i_1024_p1.png", "bench_out/big/stock_t2i_1024_p1.png",
     "docs/img/pair_textsign.png",
     "43.4 s", "199.5 s",
     m("bench_out/big/t2i_1024_p1.png", "bench_out/big/stock_t2i_1024_p1.png"),
     crops=[((100, 250, 924, 800), "zoom: signage / text rendering")])

# 3) portrait — fine skin/detail
card("bench_out/big/t2i_1024_p2.png", "bench_out/big/stock_t2i_1024_p2.png",
     "docs/img/pair_portrait.png",
     "43.0 s", "241.3 s",
     m("bench_out/big/t2i_1024_p2.png", "bench_out/big/stock_t2i_1024_p2.png"),
     crops=[((256, 60, 780, 574), "zoom: face / weathered skin detail")])

# 4) stylized robot workshop
card("bench_out/big/t2i_1024_p4.png", "bench_out/big/stock_t2i_1024_p4.png",
     "docs/img/pair_workshop.png",
     "43.2 s", "229.6 s",
     m("bench_out/big/t2i_1024_p4.png", "bench_out/big/stock_t2i_1024_p4.png"))

# 5) edit — recolor (768)
card("bench_out/big/edit_768_e0.png", "bench_out/big/stock_edit_768.png",
     "docs/img/pair_edit.png",
     "35.3 s", "163.2 s",
     m("bench_out/big/edit_768_e0.png", "bench_out/big/stock_edit_768.png"))

# 6) the speed pair — teapot at 512 (9.7x)
card("bench_out/big/t2i_512_p0.png", "bench_out/big/stock_t2i_512_p0.png",
     "docs/img/pair_teapot512.png",
     "18.9 s", "183.1 s",
     m("bench_out/big/t2i_512_p0.png", "bench_out/big/stock_t2i_512_p0.png"))

print("done")