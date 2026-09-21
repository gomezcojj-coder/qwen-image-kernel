# Contiguous preallocated prefix KV cache for Qwen-Image 2.1 edits.
#
# Why: the editing path's prefix KV cache is large (a 1024^2 reference is 4096
# latent tokens -> ~2.1 GB of bf16 cache across 32 layers). Allocated as 64
# separate per-layer tensors *after* the text-encoder's 8.75 GB alloc/free
# cycle, it lands in a fragmented VRAM pool; under Windows WDDM the attention
# kernel then reads it at a fraction of normal bandwidth (observed 6.5x step
# slowdown). Allocating the whole cache as ONE slab up front - before the big
# model weights - guarantees a contiguous, TLB-friendly working set.
#
# Optional: store V in fp8e4m3 with per-head scales (K stays bf16 - post-RoPE
# keys are more sensitive). Halves V bandwidth and shrinks the slab; measured
# by the vLLM recipe at ~41 dB vs BF16 caches (V-only).

from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0


class _LayerView:
    """Per-layer store/get interface compatible with the stock cache."""

    def __init__(self, cache: "ContiguousKVCache", layer: int):
        self._c = cache
        self._layer = layer

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        # k, v: [B, P, H, D] (post-RoPE). Backing layout: [B, H, P, D] so each
        # head's token sequence is dense for the attention kernel's tile reads.
        P = k.shape[1]
        kt = k.transpose(1, 2)  # [B, H, P, D]
        self._c.k_buf[self._layer, :, :, :P, :].copy_(kt)
        if self._c.v_fp8:
            vt = v.transpose(1, 2).float()  # [B, H, P, D]
            scale = vt.abs().amax(dim=(2, 3)).float() / FP8_E4M3_MAX
            scale = scale.clamp_min(1e-12)
            vq = (vt / scale[:, :, None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            self._c.v_buf[self._layer, :, :, :P, :].copy_(vq)
            self._c.v_scale[self._layer].copy_(scale)
        else:
            self._c.v_buf[self._layer, :, :, :P, :].copy_(v.transpose(1, 2))
        self._c.lengths[self._layer] = P

    def get(self):
        """Returns (k, v) as [B, P, H, D] views (strided; kernel is stride-agnostic)."""
        P = int(self._c.lengths[self._layer])
        k = self._c.k_buf[self._layer, :, :, :P, :].transpose(1, 2)
        if self._c.v_fp8:
            v = self._c.v_buf[self._layer, :, :, :P, :].transpose(1, 2)
        else:
            v = self._c.v_buf[self._layer, :, :, :P, :].transpose(1, 2)
        return k, v

    def views(self):
        """(k, v, v_scale_or_None) for the kernel wrapper."""
        P = int(self._c.lengths[self._layer])
        k = self._c.k_buf[self._layer, :, :, :P, :].transpose(1, 2)
        v = self._c.v_buf[self._layer, :, :, :P, :].transpose(1, 2)
        scale = self._c.v_scale[self._layer] if self._c.v_fp8 else None
        return k, v, scale


class ContiguousKVCache:
    """Single-slab prefix KV cache: one allocation for all layers.

    K layout: [L, B, H, P_max, D] bf16. V layout: same (bf16) or fp8e4m3 with
    per-head fp32 scales [L, B, H].
    """

    def __init__(
        self,
        num_layers: int,
        batch: int,
        prefix_max: int,
        num_heads: int,
        head_dim: int,
        v_fp8: bool = False,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.v_fp8 = v_fp8
        self.num_layers = num_layers
        self.capacity = prefix_max
        self.batch_size = batch
        self.k_buf = torch.empty(num_layers, batch, num_heads, prefix_max, head_dim,
                                 dtype=dtype, device=device)
        v_dtype = torch.float8_e4m3fn if v_fp8 else dtype
        self.v_buf = torch.empty(num_layers, batch, num_heads, prefix_max, head_dim,
                                 dtype=v_dtype, device=device)
        self.v_scale = torch.empty(num_layers, batch, num_heads, dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_layers, dtype=torch.int64, device=device)
        self.layers = [_LayerView(self, i) for i in range(num_layers)]

    def get_layer(self, i: int) -> _LayerView:
        return self.layers[i]