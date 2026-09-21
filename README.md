# Qwen-Image-2.1 Kernel Runtime

I wanted to run [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
on my own 12 GB card, and none of the published paths fit: BF16 needs ~34 GB,
SGLang stops at the 4090 with offload, and the 12 GB advice was to drop to 3/4-bit
GGUF quants. So I built a kernel runtime that gets the full model onto a 12 GB
Blackwell card at FP8 accuracy, on Windows, without torch.compile.

Same prompt and seed, this runtime (left) vs stock diffusers with sequential
offload (right), both 1024px / 40 steps:

<p align="center">
  <img src="docs/img/anime_kernel_fp8.png" width="44%"> <img src="docs/img/anime_stock_bf16.png" width="44%">
</p>
<p align="center"><em>Kernel: 74.7 s cold, 7.6 GiB VRAM. Stock: 290.9 s. SSIM 0.90, differences are fine-texture level.</em></p>

| 40 steps, CFG off, seed 42 | wall time | vs stock | VRAM | accuracy vs BF16 |
|---|---|---|---|---|
| text-to-image 1024px | **43.8 s** | 220.7 s (**5.0x**) | 7.6 GiB | 29.8 dB PSNR |
| text-to-image 512px | **18.9 s** | 183.1 s (**9.7x**) | 8.9 GiB | 28.2-29.9 dB |
| image edit (768 ref) | **35.3 s** | 163.2 s (**4.6x**) | | **36.5 dB PSNR** (matched) |
| determinism (same seed) | **bit-exact** | - | - | - |

Full benchmark: [RESULTS.md](RESULTS.md). How this compares to vLLM-Omni, SGLang,
LightX2V, ComfyUI quants and GGUF: [COMPARISON.md](COMPARISON.md).

## Install

Needs an NVIDIA GPU with CUDA 12.8+ (RTX 5070 / sm_120 tested, Ada sm_89 likely
works), 12 GB VRAM, 32 GB RAM, ~35 GB free disk, Python 3.11, Windows. Linux
should work but I have not tested it.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# one time: quantize DiT + text-encoder linears to FP8 (~10 min, ~15.5 GB on disk)
python tools\build_fp8_caches.py
```

## Usage

```powershell
# text-to-image (1024px default, 40 steps)
python generate.py --prompt "a ceramic teapot on a wooden table, morning light" --seed 42

# image editing, one or more references (RGBA supported)
python generate.py --prompt "Repaint the teapot with a glossy cobalt blue glaze" --image teapot.png
python generate.py --prompt "Combine these images: ..." --image a.png b.png
```

There is also a local web UI: `python webui\server.py`, then open
http://127.0.0.1:8189. It keeps the runtime resident, has generate/edit tabs,
drag-and-drop references, live step progress, and a gallery under
`webui_output/`. Generations run one at a time through a job queue.

For scripts: `qwen_image_kernel.runtime.KernelRuntime` with `generate(prompt,
image=[...], output_resolution=1024, num_inference_steps=40, seed=...)`. The
runtime caches prompt embeddings and keeps the DiT resident between calls.

## How it works

**Exact block-causal attention** (`qwen_image_kernel/triton_attention.py`).
Qwen-Image 2.1's attention mask is `(q_idx >= kv_idx) or same_image_block`.
The observation this repo is built on: that mask collapses to a per-row prefix
interval. Text tokens attend `[0, q]`, image-block tokens attend `[0, block_end)`,
decode-mode target tokens attend everything. So the whole mask is just a per-row
causal-length array, and one online-softmax flash kernel reproduces it exactly.
That matters on Windows because the stock fast path needs `flex_attention` +
torch.compile, and the stock fallback is a multi-pass SDPA decomposition. The
same kernel also reads the cross-step prefix KV cache directly (cached prefix +
fresh target K/V) instead of `torch.cat`-ing 32 layers of cache every step.

**Fused elementwise kernels** (`triton_fused.py`): per-head RMSNorm+RoPE,
LayerNorm+modulation (with the causal-condition t/t=0 row selection),
gate+residual, SwiGLU. Each replicates the stock bf16 rounding points.

**FP8 linears** (`fp8.py`): fp8e4m3 weights quantized once offline, dynamic
per-tensor activation scales, GEMMs through `torch._scaled_mm`. I probed sm_120
with cu130 first: tensorwise FP8 works and is ~1.95x faster than BF16 at the
DiT's shapes; rowwise/blockwise FP8 and MXFP8 are not usable through
`torch._scaled_mm` on this device (`tools/probe_fp8.py`).

**Contiguous prefix-KV slab** (`kv_cache.py`): the edit path's prefix cache (up to
~2 GB) is one preallocated slab, allocated before the DiT loads and reused across
generations. A cache scattered across a fragmented pool made the attention kernel
read at a fraction of bandwidth under WDDM (1024px-reference edits: 453 s before,
83.6 s after). The V half of the cache is stored in fp8 with per-head scales;
K stays bf16.

**12 GB orchestration** (`runtime.py`): text encoder (FP8) -> encode -> unload;
DiT (FP8) -> denoise; VAE (fp32) -> tiled decode. Weights load through a
meta-device skeleton from the FP8 cache, so no BF16 copy ever exists.

## Image editing

Pass one or more references (`--image ref.png`, RGBA supported). The prompt and
condition images go through the FP8 Qwen3-VL vision tower, the references are
VAE-encoded in fp32 into prefix latents, and denoising runs over
`[text, cond blocks..., target]` with the contiguous prefix-KV slab.

| check | result |
| --- | --- |
| recolor vs stock BF16 (matched 768 ref + output) | **36.5 dB PSNR** |
| recolor wall time (768 ref) | 163.2 s stock, **35.3 s** here (4.6x) |
| recolor, 1024px reference | 223.4 s stock, **83.6 s** here (was 453 s before the slab fix) |
| RGBA compositing | `docs/img/edit_rgba_beach.png` |
| multi-image | `docs/img/edit_multi.png` |
| reference budget | auto: 768px (1 image), 640px (2), 512px (3+); override with `ref_resolution=...` |
| fp8 cache-V accuracy cost | none measurable (36.5 dB with fp8-V vs 39.3 dB bf16 cache; within run-to-run noise) |

## Accuracy: what FP8 actually costs

Every number in this repo is measured against the stock BF16 pipeline at identical
seed. Matched pairs (full frames + zoom crops) are in `docs/img/`, per-pair
metrics in `bench_out/pair_metrics.json`:

| pair | PSNR | SSIM |
| --- | --- | --- |
| edit recolor @768 | 36.51 dB | 0.9708 |
| teapot @512 | 28.15 dB | 0.9532 |
| valley @1024 | 27.49 dB | 0.9304 |
| anime girl @1024 | 25.04 dB | 0.9012 |
| workshop @1024 | 22.40 dB | 0.9048 |
| text sign @1024 | 21.07 dB | 0.8403 |
| portrait @1024 | 21.07 dB | 0.7803 |

Two things worth saying plainly. First, kernel-vs-stock runs are not
seed-matched reproductions of each other: FP8 rounding noise through 40 denoising
steps changes fine texture, and it can occasionally flip something visible (in the
portrait pair the sampled face identity differs). Composition, style and prompt
adherence held in every pair. Second, the attention kernel itself is exact, only
the GEMMs are quantized; if you need BF16 outputs, `--no-fp8` runs the same
kernels unquantized (needs 24 GB for the BF16 DiT).

## Tests

`tests/` compares every kernel against the stock implementation:

| test | max abs diff |
| --- | --- |
| attention vs stock multi-pass SDPA (t2i, edit, padding, batch=2) | <= 5e-4 |
| attention vs fp32 torch reference (decode + prefix cache) | <= 2.4e-4 |
| fused RMSNorm+RoPE vs stock | 4.9e-4 |
| fused LN+modulate / gate+residual / SwiGLU vs stock | <= 3.1e-2 (bf16 rounding points) |
| full kernel block vs stock block (prefill + extract + decode) | 1.6e-2; cached K/V bit-exact |

## Notes and troubleshooting

* First generation is slow (~80 s): Triton JIT + cuBLASLt heuristics + model
  loads. Later generations reuse the resident DiT and KV slab.
* `Rowwise scaling is not currently supported on your device` is expected on
  sm_120; the runtime uses tensorwise scaling. `tools/probe_fp8.py` re-probes.
* Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; cuMemSetAccess
  fails under Windows WDDM.
* Reference images and VRAM: every 16px latent prefix token costs ~512 KB of KV
  cache across 32 layers. Default budgets are 768px (1 image), 640px (2), 512px
  (3+). 1024px references work (1.2-2.2 s/step) but sit close to the 12 GB limit.
* Disk: ~31 GB BF16 checkpoint, ~15.5 GB FP8 caches.

## License and credits

Code: Apache-2.0, derived from [diffusers](https://github.com/huggingface/diffusers).
The model weights are Qwen-Image-2.1 under the `qwen-research` license
(non-commercial); check the model card before any commercial use.
References: the diffusers implementation (PR #14804) and the
[vLLM-Omni recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1), whose GB200
measurements (26.1-29.7 dB for FP8) set the accuracy bar this repo compares against.

## Limitations

* Tested on one GPU (RTX 5070, sm_120) on Windows 11. Other Blackwell/Ada cards
  and Linux should work but are unverified. Ampere lacks FP8 tensor cores.
* FP8 uses per-tensor scales; cuBLASLt on consumer Blackwell does not expose
  rowwise/blockwise FP8 or MXFP4 through `torch._scaled_mm` yet. When it does,
  accuracy and speed both improve.
* The FP8-only ablation cannot be compared against unquantized kernels on 12 GB
  (the BF16 DiT alone is 14.2 GB).
* For throughput serving use vLLM-Omni. This is a single-user local runtime.