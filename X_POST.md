# X / Twitter announcement draft

## Main post (single tweet, under 280 chars)

---
Qwen-Image-2.1 (7B) doesn't fit a 12 GB card — so I built a kernel runtime that makes it fit.

FP8 W8A8 + exact Triton block-causal attention, no torch.compile, Windows.

1024px/40 steps in 44 s (5x stock), edits in 35 s, 7.6 GiB VRAM, bit-exact determinism.

https://github.com/YOURNAME/qwen-image-kernel
---

## Thread version

**1/**
Qwen-Image-2.1 is a great 7B image model (native RGBA, editing, text rendering) — but BF16 it needs ~34 GB. Every published path stops at 24 GB cards or drops to 3/4-bit GGUF quants with visible quality loss.

I built a runtime for the 12 GB class. Here's what it does:

**2/**
The trick #1 — the model's "block-causal" attention mask collapses to a simple per-row prefix interval. One exact Triton flash-attention kernel replaces both flex_attention (needs torch.compile, painful on Windows) and the multi-pass SDPA fallback.

**3/**
The trick #2 — FP8 W8A8 linears via cuBLASLt. Probed on sm_120: rowwise/blockwise FP8 and MXFP8 don't work through torch._scaled_mm, but tensorwise does — 1.95x faster than BF16 at the DiT's shapes. Measured 29.8 dB (t2i) / 36.5 dB (edits) vs BF16.

**4/**
The trick #3 — the edit path's prefix KV cache is preallocated as one contiguous slab (V in FP8 with per-head scales). A scattered cache under WDDM made attention read at 1/6 bandwidth — 1024px-reference edits went from 453 s to 84 s.

**5/**
Numbers on an RTX 5070 12 GB, 1024px, 40 steps, vs stock diffusers (sequential offload, same seed):
- 512px: 18.9 s vs 183 s (9.7x)
- 1024px: 43.8 s vs 221 s (5.0x)
- edits (768 ref): 35.3 s vs 163 s (4.6x)
- bit-exact determinism, 7.6 GiB VRAM peak

**6/**
Accuracy is measured, not vibes: unit tests pin the attention kernel to the stock implementation (exact), only the GEMMs are quantized. Full benchmark + per-image comparisons vs stock BF16 in the repo.

**7/**
Limits (honest): tested on one GPU (5070, sm_120), Windows-first, tensorwise FP8 only (cuBLASLt on consumer Blackwell rejects rowwise/blockwise). Weights are qwen-research licensed — non-commercial.

Code + benchmarks + install: https://github.com/YOURNAME/qwen-image-kernel

## Shorter alt (if you want one image)

---
Made Qwen-Image-2.1 run on a 12 GB card.

FP8 W8A8 + an exact Triton block-causal attention kernel (no torch.compile, Windows-native).

1024px in 44 s — 5x faster than stock diffusers — at 7.6 GiB VRAM, 29.8 dB vs BF16, bit-exact.

Repo + benchmarks:
https://github.com/YOURNAME/qwen-image-kernel
---