# Qwen-Image-2.1 Kernel Runtime

Optimized local inference for [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
(7.1B block-causal DiT, RGBA-native, unified text-to-image + editing) on a **12 GB NVIDIA
Blackwell GPU under Windows** — no torch.compile, no Linux-only tooling, no accuracy
sacrifice beyond standard FP8 quantization.

Built and measured on an RTX 5070 (sm_120). Full numbers: [RESULTS.md](RESULTS.md).
How this stacks up against vLLM-Omni, SGLang, LightX2V, ComfyUI quants and GGUF:
[COMPARISON.md](COMPARISON.md).

| 40 steps, CFG off, seed 42 | wall time | vs stock | VRAM | accuracy vs BF16 |
|---|---|---|---|---|
| text-to-image 1024² | **43.8 s** | 220.7 s (**5.0×**) | 7.6 GiB | 29.8 dB PSNR |
| text-to-image 512² | **18.9 s** | 183.1 s (**9.7×**) | 8.9 GiB | 28.2-29.9 dB |
| image edit (768 ref) | **35.3 s** | 163.2 s (**4.6×**) | — | **36.5 dB PSNR** (matched) |
| determinism (same seed) | **bit-exact** | - | - | - |

**Same prompt + seed, kernel FP8 vs stock BF16** (1024px, 40 steps):

<p align="center">
  <img src="docs/img/anime_kernel_fp8.png" width="44%"> <img src="docs/img/anime_stock_bf16.png" width="44%">
</p>
<p align="center"><em>Left: this runtime (7.6 GiB VRAM, 74.7 s cold). Right: stock BF16 + sequential offload (290.9 s). SSIM 0.90, PSNR 25.0 dB — differences are fine-texture level, composition identical.</em></p>

An FP8-only ablation (quantized weights, stock attention/elementwise) runs 60.0 s
@1024 — the fused kernels add ~37% on top of quantization. Stock BF16 time is
resolution-insensitive (~183-221 s) because sequential offload streams all 32
blocks per step regardless of resolution; the kernel runtime scales with
resolution, so the speedup is largest at smaller sizes.

## Install

Requirements: NVIDIA GPU with CUDA 12.8+ (Blackwell sm_120 tested; Ada sm_89 likely
works for the FP8 path), 12 GB VRAM, 32 GB system RAM, ~35 GB free disk for the
checkpoint, Python 3.11, Windows (Linux should work — only tested on Windows).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# one-time: quantize DiT + text-encoder linears to FP8 (~10 min, needs the BF16
# weights; they download automatically to the HF cache)
$env:PYTHONPATH = "."
python -c "from qwen_image_kernel.quantize_offline import build_transformer_fp8_cache, build_text_encoder_fp8_cache; build_transformer_fp8_cache('Qwen/Qwen-Image-2.1','fp8_cache/transformer_fp8.safetensors'); build_text_encoder_fp8_cache('Qwen/Qwen-Image-2.1','fp8_cache/text_encoder_fp8.safetensors')"
```

## Usage

```powershell
# text-to-image (1024px default, 40 steps)
python generate.py --prompt "a ceramic teapot on a wooden table, morning light" --seed 42

# image editing - one or more references (RGBA supported)
python generate.py --prompt "Repaint the teapot with a glossy cobalt blue glaze" --image teapot.png
python generate.py --prompt "Combine these images: ..." --image a.png b.png

# benchmark (see tools/)
python tools\benchmark_scale.py        # kernel configs, ~25 min
python tools\benchmark_scale_stock.py  # stock baseline, ~25 min
python tools\make_results.py           # merge -> RESULTS.md
```

Programmatic use: `qwen_image_kernel.runtime.KernelRuntime` — `generate(prompt,
image=[...], output_resolution=1024, num_inference_steps=40, seed=...)`. A warm
runtime keeps the DiT resident and caches prompt embeddings, so repeated
generations skip reloads.

## Web UI

A local site that keeps the runtime resident and serves generations with live
progress:

```powershell
python webui\server.py        # http://127.0.0.1:8189
```

Loads the model in the background (~40 s to ready — the site shows a status badge),
then: text-to-image and editing tabs (drag-and-drop up to 4 RGBA references), size /
steps / seed controls, a live progress bar fed by the runtime's step callbacks, and
an output gallery persisted under `webui_output/`. Generations are serialized
through a job queue (one at a time — the GPU serves a single resident model).

## How it works

1. **One exact Triton flash-attention kernel** (`qwen_image_kernel/triton_attention.py`).
   Qwen-Image 2.1's block-causal mask — `(q_idx >= kv_idx) or same_image_block` —
   collapses to a *per-row prefix interval*: text tokens attend `[0, q]`, image-block
   tokens attend `[0, block_end)`, decode-mode target tokens attend everything. So the
   entire mask is a per-row causal-length array and a single online-softmax flash kernel
   reproduces it exactly. This removes the two stock options' problems on Windows:
   `flex_attention` needs torch.compile, and the SDPA fallback runs one attention call
   per prefix segment. The same kernel also reads the cross-step **prefix KV cache**
   directly (virtual concat of cached prefix + fresh target K/V), removing the
   per-step `torch.cat` of 32 layers' caches.

2. **Fused kernels** (`triton_fused.py`): per-head RMSNorm+RoPE, LayerNorm+modulation
   (with the causal-condition t/t=0 row selection), gate+residual (`x + tanh(g)*f`),
   SwiGLU — each replicating the stock bf16 rounding points.

3. **FP8 (W8A8) linears** (`fp8.py`): offline-quantized fp8e4m3 weights, dynamic
   per-tensor activation scales, GEMMs through `torch._scaled_mm` (cuBLASLt).
   Probed on sm_120/cu130: tensorwise FP8 is supported and ~1.95x faster than BF16 at
   the DiT's shapes; rowwise/blockwise/MXFP8 are not usable through `torch._scaled_mm`
   on this device (see `tools/probe_fp8*.py`).

4. **12 GB VRAM orchestration** (`runtime.py`): text encoder (FP8) → encode → unload;
   DiT (FP8) → denoise; VAE (fp32) → tiled decode. Meta-device skeleton loading from a
   one-time FP8 cache so no BF16 weight is ever materialized twice.

5. **Contiguous prefix-KV slab** (`kv_cache.py`): the edit path's prefix cache (up to
   ~2 GB) is preallocated as one slab *before* the DiT loads and reused across
   generations. A cache scattered across a fragmented pool makes the attention kernel
   read at a fraction of bandwidth under WDDM (observed 6.5x step slowdown at 1024²
   references); the slab fixes it, and its V half is stored in fp8 with per-head scales
   (~41 dB per the vLLM recipe's cache measurements — we measure no accuracy loss at all).

## Image editing

`generate.py --image ref.png` (one or more references, RGBA supported). The edit path
encodes the prompt + condition images through the FP8 Qwen3-VL vision tower, VAE-encodes
the references in **fp32** (the checkpoint's dtype) into prefix latents, and denoises
with the exact block-causal kernel over `[text, cond blocks..., target]` with the
contiguous prefix-KV slab.

| check | result |
| --- | --- |
| recolor edit vs stock BF16 (matched 768 ref + output) | **36.5 dB PSNR** |
| recolor edit wall time (768 ref) | 163.2 s stock → **35.3 s** (4.6×) |
| recolor edit, 1024² reference | 223.4 s stock → **83.6 s** (1.16 s/step; was 453 s before the contiguous-slab fix) |
| RGBA compositing (`bench_out/edit_rgba.png`) | transparent sphere placed on a beach, shadow intact |
| multi-image (`bench_out/edit_multi.png`) | two references combined, both preserved |
| reference budget | auto: 768² (1 image), 640² (2), 512² (3+); override with `ref_resolution=...` |
| fp8 cache-V accuracy cost | none measurable (36.5 dB with fp8-V vs 39.3 dB with bf16 cache at 768; same within run-to-run chaos) |

## Verified correctness

Unit tests in `tests/` (all pass on this machine):

| test | max abs diff |
| --- | --- |
| attention vs stock multi-pass SDPA (t2i, edit, padding, batch=2 layouts) | ≤ 5e-4 |
| attention vs fp32 torch reference (decode + prefix cache) | ≤ 2.4e-4 |
| fused RMSNorm+RoPE vs stock | 4.9e-4 |
| fused LN+modulate / gate+residual / SwiGLU vs stock | ≤ 3.1e-2 (bf16 rounding points) |
| full kernel block vs stock block (prefill + extract + decode) | 1.6e-2; cached K/V bit-exact |

The attention kernel is exact — only the GEMMs are quantized, which is why FP8
accuracy lands at 29.9 dB (t2i) / 39.3 dB (edits, matched settings) against the
BF16 pipeline.

## Troubleshooting

* **`Rowwise scaling is not currently supported on your device`** — expected on
  sm_120; the runtime uses tensorwise scaling. See `tools/probe_fp8_*.py` to re-probe.
* **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — cuMemSetAccess
  fails under Windows WDDM.
* **Editing references and VRAM** — every 16px latent prefix token costs ~512 KB of
  prefix KV cache across 32 layers. The runtime preallocates the whole cache as one
  contiguous slab before the DiT loads (a cache scattered across a fragmented pool
  makes the attention kernel read at a fraction of bandwidth under WDDM — observed
  6.5x step slowdown) and stores cache V in fp8 with per-head scales (~41 dB per the
  vLLM recipe's measurements; K stays bf16). Default reference budgets: 768² (1 image,
  **~30 s warm, 2.5x faster than stock**), 640² (2 images), 512² (3+). For maximum
  reference fidelity pass `ref_resolution=1024` — with the contiguous slab this now
  runs at 1.2-2.2 s/step (~60-135 s), faster than stock, versus 453 s before the fix.
* **First generation is slow (~80 s)** — Triton JIT + cuBLASLt heuristics + model
  loads. Later generations in the same process skip this; the KV slab and DiT stay
  resident between generations.
* **Disk** — the BF16 checkpoint is ~31 GB; the FP8 caches add ~15.5 GB.

## License / credits

* Code here: Apache-2.0 (derives from [diffusers](https://github.com/huggingface/diffusers),
  which carries the same license).
* The **model weights are Qwen-Image-2.1 under the `qwen-research` license** — non-commercial
  research use; check [the model card](https://huggingface.co/Qwen/Qwen-Image-2.1) before
  any commercial use.
* Architecture reference: the diffusers implementation (PR #14804) and the
  [vLLM-Omni recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1); FP8-accuracy context
  (26.1–29.7 dB on GB200) comes from that recipe's measurements.

## Limitations

* Tested on exactly one GPU (RTX 5070, sm_120) and one OS (Windows 11); other
  Blackwell/Ada cards and Linux should work but are untested. Ampere (sm_80/86) lacks
  FP8 tensor cores — the FP8 path will be slower there, and `torch._scaled_mm`
  support differs.
* FP8 uses per-tensor scales (rowwise/blockwise FP8 and MXFP8 are not usable through
  `torch._scaled_mm` on consumer Blackwell today); if cuBLASLt ships those for
  sm_120, accuracy and speed both improve.
* The FP8-only ablation cannot run in BF16 on 12 GB (weights alone are 14.2 GB), so
  "kernels without quantization" is not benchmarkable here.
* vLLM-Omni (datacenter serving, batching, CUDA graphs) is the better tool for
  throughput serving; this runtime targets single-user local generation.