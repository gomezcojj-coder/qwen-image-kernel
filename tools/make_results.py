# Merge the kernel + stock benchmark JSONs into RESULTS.md tables and PSNR pairs.
import json
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.chdir(Path(__file__).resolve().parent.parent)
from qwen_image_kernel.metrics import image_psnr

OUT = "bench_out/big"
k = json.load(open("bench_out/kernel_bench.json"))
s = json.load(open("bench_out/stock_bench.json"))
try:
    k2 = json.load(open("bench_out/kernel_bench2.json"))
    k["t2i"] = k["t2i"] + k2.get("t2i_fp8only", [])
    k["edit"] = k["edit"] + k2.get("edit", [])
except FileNotFoundError:
    k2 = None

# ---------------- t2i tables ----------------
def agg_t2i(cfg, res):
    rows = [r for r in k["t2i"] if r["cfg"] == cfg and r["res"] == res]
    if not rows:
        return None
    return {
        "n": len(rows),
        "total_s": sum(r["total_s"] for r in rows) / len(rows),
        "step_s": sum(r["mean_step_s"] for r in rows) / len(rows),
        "vram_gib": max(r["vram_gib"] for r in rows),
    }

a = {res: agg_t2i("A", res) for res in (512, 768, 1024)}
c1024 = agg_t2i("C", 1024)
b = {}
for res in (512, 768, 1024):
    rows = [r for r in s["t2i"] if r["cfg"] == "B" and r["res"] == res]
    if rows:
        b[res] = {
            "total_s": sum(r["total_s"] for r in rows) / len(rows),
            "step_s": sum(r["mean_step_s"] for r in rows) / len(rows),
            "vram_gib": max(r["vram_gib"] for r in rows),
        }

lines = []
lines.append("# Qwen-Image-2.1 Kernel Runtime - Benchmark Results\n")
lines.append(f"GPU: {k['meta']['gpu']} | torch {k['meta']['torch']} | Windows 11 | 40 steps, CFG off, seed 42\n")
lines.append("Timed runs average over 5 prompts (kernel) / 2 prompts (stock); each timed run is "
             "preceded by an untimed warmup at the same shape. VRAM = torch.cuda.max_memory_allocated() "
             "during the denoise loop. The FP8-only ablation keeps quantized weights but uses the stock "
             "attention/elementwise ops.\n")

lines.append("## Text-to-image performance\n")
lines.append("| resolution | stock BF16 (offload) | kernels + FP8 | speedup | FP8 only (no fused kernels) | VRAM peak (kernels+FP8) |")
lines.append("|---|---|---|---|---|---|")
for res in (512, 768, 1024):
    bb, aa = b[res], a[res]
    extra = f"{c1024['total_s']:.1f} s ({c1024['step_s']:.2f} s/step)" if res == 1024 else "-"
    lines.append(f"| {res}px | {bb['total_s']:.1f} s ({bb['step_s']:.2f} s/step) | "
                 f"{aa['total_s']:.1f} s ({aa['step_s']:.2f} s/step) | **{bb['total_s']/aa['total_s']:.2f}x** | "
                 f"{extra} | {aa['vram_gib']:.2f} GiB |")

# per-prompt detail at 1024
lines.append("\n### Per-prompt detail @1024 (config A)\n")
lines.append("| prompt | total | s/step | VRAM |")
lines.append("|---|---|---|---|")
names = ["teapot (photoreal)", "bookshop sign (text render)", "fisherman (portrait)", "mountain valley (landscape)", "robot workshop (stylized)"]
for r in k["t2i"]:
    if r["cfg"] == "A" and r["res"] == 1024:
        lines.append(f"| {names[r['prompt']]} | {r['total_s']:.1f} s | {r['mean_step_s']:.2f} s | {r['vram_gib']:.2f} GiB |")

# ---------------- accuracy ----------------
lines.append("\n## Accuracy (kernel FP8 vs stock BF16, identical seeds/settings)\n")
lines.append("| task | PSNR |")
lines.append("|---|---|")
for res in (512, 768, 1024):
    for i in (0, 3):
        try:
            p = image_psnr(f"{OUT}/t2i_{res}_p{i}.png", f"{OUT}/stock_t2i_{res}_p{i}.png")
            lines.append(f"| t2i {res}px, prompt {i} | {p:.2f} dB |")
        except Exception:
            pass
lines.append(f"| determinism (same config, same seed, rerun) | {'infinite (bit-exact)' if k.get('determinism_bitexact') else 'see json'} |")

# edits
lines.append("\n### Image editing\n")
lines.append("| task | kernel total | stock total | PSNR (matched) |")
lines.append("|---|---|---|---|")
ke = {(e["ref_res"], e["edit"]): e for e in k["edit"] if e["cfg"] == "A"}
se = {(e["ref_res"], e["edit"]): e for e in s["edit"]}
for key, label in [((768, 0), "recolor, 768 ref"), ((1024, 0), "recolor, 1024 ref")]:
    if (key in ke) and (key in se):
        kk, ss = ke[key], se[key]
        pair = ("edit_768_e0.png", "stock_edit_768.png") if key[0] == 768 else ("edit_1024_e0.png", "stock_edit_1024.png")
        try:
            p = image_psnr(f"{OUT}/{pair[0]}", f"{OUT}/{pair[1]}")
        except Exception:
            p = float("nan")
        lines.append(f"| {label} | {ke[key]['total_s']:.1f} s ({ke[key]['mean_step_s']:.2f} s/step) | "
                     f"{se[key]['total_s']:.1f} s | **{p:.2f} dB** |")

open("RESULTS.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
