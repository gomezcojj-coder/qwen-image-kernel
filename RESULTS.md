# Qwen-Image-2.1 Kernel Runtime - Benchmark Results

GPU: NVIDIA GeForce RTX 5070 | torch 2.14.0+cu130 | Windows 11 | 40 steps, CFG off, seed 42

Timed runs average over 5 prompts (kernel) / 2 prompts (stock); each timed run is preceded by an untimed warmup at the same shape. VRAM = torch.cuda.max_memory_allocated() during the denoise loop. The FP8-only ablation keeps quantized weights but uses the stock attention/elementwise ops.

## Text-to-image performance

| resolution | stock BF16 (offload) | kernels + FP8 | speedup | FP8 only (no fused kernels) | VRAM peak (kernels+FP8) |
|---|---|---|---|---|---|
| 512px | 183.1 s (2.67 s/step) | 18.9 s (0.21 s/step) | **9.71x** | - | 8.92 GiB |
| 768px | 188.3 s (3.49 s/step) | 26.0 s (0.51 s/step) | **7.24x** | - | 7.26 GiB |
| 1024px | 220.7 s (3.90 s/step) | 43.8 s (0.94 s/step) | **5.04x** | 60.0 s (1.22 s/step) | 7.63 GiB |

### Per-prompt detail @1024 (config A)

| prompt | total | s/step | VRAM |
|---|---|---|---|
| teapot (photoreal) | 45.2 s | 0.97 s | 7.62 GiB |
| bookshop sign (text render) | 44.0 s | 0.94 s | 7.63 GiB |
| fisherman (portrait) | 43.0 s | 0.92 s | 7.63 GiB |
| mountain valley (landscape) | 43.9 s | 0.94 s | 7.62 GiB |
| robot workshop (stylized) | 43.2 s | 0.92 s | 7.62 GiB |

## Accuracy (kernel FP8 vs stock BF16, identical seeds/settings)

| task | PSNR |
|---|---|
| t2i 512px, prompt 0 | 28.15 dB |
| t2i 512px, prompt 3 | 29.94 dB |
| t2i 768px, prompt 0 | 22.22 dB |
| t2i 768px, prompt 3 | 36.50 dB |
| t2i 1024px, prompt 0 | 29.83 dB |
| t2i 1024px, prompt 3 | 27.49 dB |
| determinism (same config, same seed, rerun) | infinite (bit-exact) |

### Image editing

| task | kernel total | stock total | PSNR (matched) |
|---|---|---|---|
| recolor, 768 ref | 35.3 s (0.52 s/step) | 163.2 s | **36.51 dB** |
| recolor, 1024 ref | 83.6 s (1.16 s/step) | 223.4 s | **29.78 dB** |
