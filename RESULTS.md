# Qwen-Image-2.1 Kernel Runtime - Benchmark Results

GPU: NVIDIA GeForce RTX 5070 | torch 2.14.0+cu130 | Windows 11 | 40 steps, CFG off, seed 42

Timed runs average over 5 prompts (kernel) / 2 prompts (stock); each timed run is preceded by an untimed warmup at the same shape. VRAM = torch.cuda.max_memory_allocated() during the denoise loop. The FP8-only ablation keeps quantized weights but uses the stock attention/elementwise ops.

Config A now additionally uses: fused qkv / mlp-up FP8 GEMMs (single-rounding quantized, replacing the individual weights), one fused Triton scale+cast kernel for the fp8 activation quant (was an abs/div/clamp/cast aten chain), a uniform-L decode fast path in the exact attention kernel with exp2 softmax, and cached RoPE cos/sin tables. Semantics are unchanged; outputs remain bit-exact deterministic and PSNR vs the BF16 stock pipeline is in the same band as before (average over the 6 reference configs: 29.2 dB, was 29.0 dB).

## Text-to-image performance

| resolution | stock BF16 (offload) | kernels + FP8 (fused) | speedup | previous config A | VRAM peak (kernels+FP8) |
|---|---|---|---|---|---|
| 512px | 183.1 s (2.67 s/step) | 8.4 s (0.15 s/step) | **21.8x** | 18.9 s | 6.95 GiB |
| 768px | 188.3 s (3.49 s/step) | 17.3 s (0.36 s/step) | **10.9x** | 26.0 s | 7.17 GiB |
| 1024px | 220.7 s (3.90 s/step) | 35.4 s (0.71 s/step) | **6.2x** | 43.8 s | 7.4-8.9 GiB (prompt-length dependent) |

### Per-prompt detail @1024 (config A, fused)

| prompt | total | s/step |
|---|---|---|
| teapot (photoreal) | 32.5 s | 0.715 |
| bookshop sign (text render) | 37.8 s | 0.713 |
| fisherman (portrait) | 35.8 s | 0.713 |
| mountain valley (landscape) | 35.2 s | 0.714 |
| robot workshop (stylized) | 35.6 s | 0.714 |

Decode step profile @1024: fp8 GEMMs 65%, exact Triton attention 23% (4.67 ms/layer, ~62 TFLOPS), everything else ~12%.

## Accuracy (kernel FP8 vs stock BF16, identical seeds/settings)

### Matched-pair gallery

Labeled side-by-side cards (full frame + zoom crops) live in `docs/img/`:

| pair | PSNR | SSIM | card |
|---|---|---|---|
| edit recolor @768 | 36.51 dB | 0.9708 | `docs/img/pair_edit.png` |
| teapot @512 | 28.15 dB | 0.9532 | `docs/img/pair_teapot512.png` |
| valley @1024 | 27.49 dB | 0.9304 | - |
| anime girl @1024 | 25.04 dB | 0.9012 | `docs/img/pair_anime.png` (+ diff heatmap) |
| workshop @1024 | 22.40 dB | 0.9048 | `docs/img/pair_workshop.png` |
| text sign @1024 | 21.07 dB | 0.8403 | `docs/img/pair_textsign.png` |
| portrait @1024 | 21.07 dB | 0.7803 | `docs/img/pair_portrait.png` |

**How to read this:** kernel-vs-stock runs are *not* seed-matched reproductions of
each other: FP8 rounding noise through 40 chaotic denoising steps produces
different fine detail, and occasionally (see the portrait pair: the sampled face
identity diverges) a semantically meaningful attribute flips. Composition, style,
layout and prompt adherence are preserved in every pair; the divergence shows up
in fine texture, small signage glyphs, and rarely in subject identity. If you need
outputs bit-matched to the BF16 pipeline, run with `--no-fp8`. This is the same
phenomenon the vLLM-Omni recipe measures (26.1 dB for all-layers FP8 on GB200); our per-prompt range is 21-37 dB with edits highest (the reference conditioning
anchors the output).

| task | PSNR (config A, fused) | PSNR (previous config A) |
|---|---|---|
| t2i 512px, prompt 0 | 18.98 dB | 28.15 dB |
| t2i 512px, prompt 3 | 34.91 dB | 29.94 dB |
| t2i 768px, prompt 0 | 24.23 dB | 22.22 dB |
| t2i 768px, prompt 3 | 36.89 dB | 36.50 dB |
| t2i 1024px, prompt 0 | 27.41 dB | 29.83 dB |
| t2i 1024px, prompt 3 | 32.60 dB | 27.49 dB |
| average over the 6 configs | **29.17 dB** | 29.02 dB |
| determinism (same config, same seed, rerun) | infinite (bit-exact) | infinite (bit-exact) |

The per-prompt spread is the same FP8-noise lottery as before: each quantization
path re-rolls the chaotic fine-detail noise, so individual pairs move up and down
(the fused 512px p0 teapot is semantically identical to the stock image; its
18.98 dB is fine-texture divergence - it sits ~19.6 dB from the previous kernel's
output too, i.e. equidistant from both references).

### Image editing

| task | kernel total | stock total | PSNR (matched) |
|---|---|---|---|
| recolor, 768 ref | 28.1 s (0.52 s/step) | 163.2 s | **36.08 dB** |
| recolor, 1024 ref | 83.6 s (1.16 s/step) | 223.4 s | **29.78 dB** |
