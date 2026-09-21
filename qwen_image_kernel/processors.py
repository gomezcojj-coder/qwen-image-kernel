# Drop-in attention processor + fused transformer block for Qwen-Image 2.1.
#
# The processor replaces `QwenImage21AttnProcessor` with identical semantics:
#   * prefill (step 1): one exact block-causal flash-attention kernel over the
#     joint sequence, driven by per-row prefix lengths derived from the same
#     segment decomposition the stock processor consumes.
#   * decode (steps 2..40): one kernel that reads the cached prefix K/V and the
#     fresh target K/V directly (no torch.cat of 32 layers' caches per step).
#
# The fused block replaces `QwenImage21TransformerBlock`'s forward with the
# fused LayerNorm+modulation / gate+residual / SwiGLU kernels, keeping the
# identical module tree so checkpoint state dicts load unchanged.

from __future__ import annotations

import torch

from .triton_attention import block_causal_flash_attention
from .triton_fused import fused_gate_residual, fused_layer_norm_modulate, fused_rms_norm_rope, fused_swiglu


def _build_prefix_lengths(
    segments: list[tuple[int, int, bool]], prefix_len: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Per-row inclusive kv limit L: row q attends iff kv <= L[q].

    text run (s, e)      -> L[s:e] = s..e-1  (causal)
    image run (s, e)     -> L[s:e] = e-1     (bidirectional block + full prefix)
    target rows [P, S)   -> L = S-1          (full attention)
    """
    L = torch.arange(seq_len, dtype=torch.int32, device=device)
    for start, end, is_text in segments:
        if not is_text:
            L[start:end] = end - 1
    L[prefix_len:] = seq_len - 1
    return L


class QwenImage21KernelProcessor:
    """Exact block-causal attention for Qwen-Image 2.1 via one Triton kernel.

    Drop-in for `QwenImage21AttnProcessor`: same call signature, no
    flex_attention, no torch.compile, no per-segment SDPA loop, and the
    decode path reads the prefix KV cache without concatenation.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode: str | None = None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        H = attn.heads
        D = attn.to_q.out_features // H

        query = attn.to_q(hidden_states).unflatten(-1, (H, D))
        key = attn.to_k(hidden_states).unflatten(-1, (H, D))
        value = attn.to_v(hidden_states).unflatten(-1, (H, D))

        # Fused QK-RMSNorm + RoPE (stock: RMSNorm -> bf16 -> complex RoPE)
        query = fused_rms_norm_rope(query, attn.norm_q.weight, rotary_emb, attn.norm_q.eps)
        key = fused_rms_norm_rope(key, attn.norm_k.weight, rotary_emb, attn.norm_k.eps)

        prefix_len = cache_write_slice.stop if cache_write_slice is not None else None

        if kv_cache_mode == "extract" and layer_cache is not None and cache_write_slice is not None:
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())

        if segments is not None:
            # prefill over the full joint sequence (the cache may have just
            # been written; this step still attends over the fresh full K/V)
            assert key_valid is None or key_valid.dim() == 2
            prefix_end = segments[-1][1] if segments else 0
            L = _build_prefix_lengths(segments, prefix_end, S, query.device)
            out = block_causal_flash_attention(
                query, key, value, L, key_valid, cache_k=None, cache_v=None
            )
        else:
            # decode: queries are the target rows only; keys = [prefix, target]
            if layer_cache is not None:
                if hasattr(layer_cache, "views"):
                    cache_k, cache_v, v_scale = layer_cache.views()
                else:
                    cache_k, cache_v = layer_cache.get()
                    v_scale = None
                KV_TOTAL = cache_k.shape[1] + key.shape[1]
                L = torch.full((key.shape[1],), KV_TOTAL - 1, dtype=torch.int32, device=query.device)
                kv_mask = attention_mask
                if kv_mask is not None:
                    kv_mask = kv_mask.reshape(kv_mask.shape[0], -1)
                out = block_causal_flash_attention(
                    query, key, value, L, kv_mask, cache_k=cache_k, cache_v=cache_v,
                    cache_v_scale=v_scale,
                )
            else:
                # no cache: full attention over the whole sequence
                L = torch.full((key.shape[1],), key.shape[1] - 1, dtype=torch.int32, device=query.device)
                kv_mask = attention_mask
                if kv_mask is not None:
                    kv_mask = kv_mask.reshape(kv_mask.shape[0], -1)
                out = block_causal_flash_attention(query, key, value, L, kv_mask)

        out = out.flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


class QwenImage21KernelBlock(torch.nn.Module):
    """Fused replacement for QwenImage21TransformerBlock.

    Reuses the original block's submodules (identical names -> state dicts
    load unchanged) but runs the fused Triton kernels for norm/modulation and
    the SwiGLU MLP. Attention goes through the (already swapped) processor.
    """

    def __init__(self, original_block):
        super().__init__()
        self.img_norm1 = original_block.img_norm1        # LayerNorm, no affine (params unused)
        self.attn = original_block.attn
        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp
        self._eps = self.img_norm1.eps

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, S, D] bf16
        modulation: torch.Tensor,          # [B(+1), 4D]
        rotary_emb=None,
        attention_mask=None,
        target_token_mask=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ) -> torch.Tensor:
        mod_stride = modulation.stride(0)
        inner = self.attn.to_q.out_features
        # mod1.scale @ +0, mod1.gate @ +inner, mod2.scale @ +2*inner, mod2.gate @ +3*inner
        img_mod = fused_layer_norm_modulate(
            hidden_states, modulation, 0, mod_stride, target_token_mask, self._eps
        )
        attn_out = self.attn(
            hidden_states=img_mod,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = fused_gate_residual(hidden_states, attn_out, modulation, inner, target_token_mask)

        img_mod2 = fused_layer_norm_modulate(
            hidden_states, modulation, 2 * inner, mod_stride, target_token_mask, self._eps
        )
        gate = self.img_mlp.gate_layer(img_mod2)
        up = self.img_mlp.proj(img_mod2)
        mlp_mid = fused_swiglu(gate, up)
        mlp_out = self.img_mlp.out(mlp_mid)
        hidden_states = fused_gate_residual(hidden_states, mlp_out, modulation, 3 * inner, target_token_mask)
        return hidden_states


def convert_transformer_to_kernels(transformer, use_fp8: bool = False) -> int:
    """Swap stock blocks for fused kernel blocks; optionally FP8-quantize the
    block linears. Returns the number of blocks converted."""
    from .fp8 import FP8Linear

    blocks = transformer.transformer_blocks
    new_blocks = []
    for blk in blocks:
        kb = QwenImage21KernelBlock(blk)
        if use_fp8:
            for lin_name in ("to_q", "to_k", "to_v"):
                lin = getattr(kb.attn, lin_name)
                if isinstance(lin, torch.nn.Linear):
                    setattr(kb.attn, lin_name, FP8Linear.from_linear(lin))
            kb.attn.to_out[0] = FP8Linear.from_linear(kb.attn.to_out[0])
            for mlp_name in ("gate_layer", "proj", "out"):
                setattr(kb.img_mlp, mlp_name, FP8Linear.from_linear(getattr(kb.img_mlp, mlp_name)))
        kb.attn.set_processor(QwenImage21KernelProcessor())
        new_blocks.append(kb)
    transformer.transformer_blocks = torch.nn.ModuleList(new_blocks)
    return len(new_blocks)