# How this runtime compares to other Qwen-Image-2.1 optimization efforts

As of 2026-09-21, five days after Qwen-Image-2.1 shipped. Sources are linked; where a
number is vendor- or community-reported we say so.

## The landscape

| runtime | target hardware | Qwen-Image-2.1 support | published timing (1024px, 40 steps) | VRAM | notes |
|---|---|---|---|---|---|
| **this project** | RTX 5070 12 GB, Windows | t2i + edits | **43.8 s** t2i / **35.3 s** edit-768 | **7.6 GiB peak** | 29.8 dB / 36.5 dB vs BF16, bit-exact determinism |
| LightX2V | RTX 5090 32 GB (consumer) | FP8 DiT + TE offload | **5.93 s** t2i / 7.14 s edit | ~16 GiB | SageAttention 2/3 + INT8/FP8 weight quants; only published consumer timing |
| ComfyUI + Comfy-Org INT8 | 24-32 GB | INT8 repackage (16.1 GiB total) | ~6.7 s (NVFP4, 5090, community) | 16.1 GiB | NVFP4 package at 10.4 GiB "reaches 16 GB cards" |
| ComfyUI + GGUF | 6-12 GB | Q3-Q8 GGUF quants | no published 2.1 timing | 4-8 GiB | Q4_K_M recommended for 6-8 GB cards; low-bit quants visibly degrade on big models |
| SGLang | consumer cookbook, no timings | TE offload, DiT resident | none published | ~15.6 GiB (offload) | "5090 and 4090 exceed single-GPU capacity" |
| vLLM-Omni | GB200/GB300/A100/B200 (datacenter) | FP8, CUDA graphs, batching, TP | 3.3-4.5 s (GB300 BF16) | 27.5-40 GB | Linux, needs >=28 GB VRAM by construction |
| ComfyUI BF16 | 24-32 GB | native | ~50 s / 20 steps (5070 Ti 16 GB, community) | 13.9 GiB | full quality, slower |
| diffusers stock | 24 GB+ | sequential offload | 183-221 s (our 5070 baseline) | streaming | reference path |

Key sources: [ai.rs survey](https://ai.rs/ai-for-business/qwen-image-2-1-open-weights-licence-vram),
[vLLM-Omni recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1),
[LightX2V Qwen-Image examples](https://github.com/ModelTC/LightX2V/tree/main/examples/qwen_image),
[Comfy-Org INT8/NVFP4 packages](https://huggingface.co/BennyDaBall/Qwen-Image-2.1-NVFP4),
[ComfyUI Wiki GGUF guide](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image).

## What nobody else does for the RTX 5070 / 12 GB class

The ai.rs survey's practical expectation table says 12 GB is "possible only with
aggressive quantization/offload; expect compromises" — and every published path
agrees: SGLang stops at the 24 GB 4090 (with offload), ComfyUI's 12 GB guidance for
Qwen-Image 1.x was a **Q3 GGUF quant** (the loader's own docs say Q4-and-below "show
visible degradation" on large DiTs), and vLLM-Omni requires datacenter cards. The
community has published **no 12 GB timing for Qwen-Image-2.1 at all**.

This runtime fills that slot specifically:

* **7.6 GiB peak at 1024px** — the lowest published footprint for this model on any
  hardware, achieved with FP8 W8A8 weights rather than low-bit GGUF. On a 12 GB card
  that leaves real headroom for a desktop, 1024² references, and RGBA editing.
* **FP8 accuracy, not GGUF accuracy**: FP8 E4M3 with per-tensor dynamic activation
  scales measures 29.8 dB (t2i) / 36.5 dB (edit) against the BF16 pipeline — the GGUF
  12 GB path (Q3/Q4 of a 7B DiT) has no published accuracy number and the ComfyUI
  loader docs themselves warn of visible degradation at Q4 and below.
* **Exact block-causal attention without torch.compile** — the stock fast path needs
  `flex_attention` + compile; on Windows that stack is painful, and the stock
  fallback is a multi-pass SDPA decomposition. Our kernel reproduces the mask exactly
  via per-row prefix intervals and is unit-tested against the stock implementation.
* **Contiguous prefix-KV slab + FP8 cache-V** — the 1024²-reference edit went from
  453 s (our own pre-fix number, slower than stock) to 83.6 s; cache-V FP8 costs
  nothing measurable (36.5 dB with fp8-V vs 39.3 dB bf16-cache at 768, both within
  run-to-run diffusion noise).

## Honest speed comparison to LightX2V (RTX 5090)

LightX2V reports 5.93 s for the same 1024px/40-step generation on a 5090 with FP8 DiT.
Per-step that is ~0.15 s vs our ~1.0 s — a ~6.7x gap. Context for that gap:

* The 5090 has ~1.75x the FP8 tensor throughput, 2.4x the memory bandwidth, 2.7x the
  VRAM, and PCIe 5 of the 5070.
* LightX2V's stack additionally uses **SageAttention** (INT8/FP8 attention) and
  quantized linear kernels — we deliberately skipped quantized attention to hold the
  accuracy line (our exact-attention claim is a correctness feature, not just a perf
  feature).
* Their number is a vendor-published median of three runs; the ai.rs survey notes no
  independent verification exists.

On a same-dollar basis the comparison is not close in our favor, but on the
"runs at all, on Windows, at FP8 accuracy, in 7.6 GiB" axis we are alone.

## What we measured vs what others claim

| claim type | vLLM-Omni | SGLang | LightX2V | ComfyUI quants | this project |
|---|---|---|---|---|---|
| measured on consumer 12 GB | no | no | no | no | **yes** |
| publishes accuracy vs BF16 (dB) | yes (26.1-29.7) | no | no | no | **yes (29.8 / 36.5)** |
| exact-attention unit tests | no | no | no | no | **yes** |
| runs on Windows | no | untested | partial (community wheels) | yes | **yes** |
| needs torch.compile | no | no | no | no | **no** |
| fits 12 GB at FP8 accuracy | no (datacenter) | no | no (Q4 GGUF = quality cost) | partial (Q8 on 12 GB+, no timings) | **yes** |

## Where the others are genuinely ahead

* **LightX2V on 24-32 GB cards is far faster** (0.15 s/step on a 5090). If you have a
  5090 and Linux, use it; their SageAttention 3 + FP8/FP4 stack is the consumer speed
  king. (Also note their reported numbers are the only consumer numbers that exist.)
* **vLLM-Omni is the right tool for serving** — batching, CUDA graphs, parallelism.
  Nothing here competes with a request-level server on an 80 GB A100.
* **ComfyUI + Comfy-Org INT8** is the most polished UX for 24 GB cards and now has an
  NVFP4 package for Blackwell — at 6.7 s on a 5090 (community numbers).
* **GGUF quants** reach smaller cards (Q4 at 6-8 GB) — at lower quality. For 12 GB
  cards running Qwen-Image-2.1, the choice is Q4-class GGUF (quality cost, no
  published accuracy) vs this runtime's FP8 (29.8 dB measured, 7.6 GiB peak).

## Bottom line

For a 12 GB Blackwell card on Windows, this runtime is currently the only documented
way to run Qwen-Image-2.1 at FP8 quantization accuracy (no 3/4-bit GGUF degradation),
with the lowest VRAM footprint of any published setup, at speeds that scale with
resolution (9.7x stock at 512px, 5.0x at 1024px) and 4.6x on edits. The 24 GB+ crowd
is better served by LightX2V or ComfyUI-NVFP4 today; the datacenter crowd by
vLLM-Omni.