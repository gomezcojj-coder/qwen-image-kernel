# Block-causal flash attention for Qwen-Image 2.1, in Triton.
#
# Key insight
# -----------
# The diffusers reference implements the block-causal mask
#     allowed(q, kv) = (q_idx >= kv_idx) | same_image_block(q, kv)
# either as a flex_attention BlockMask (requires torch.compile) or as an exact
# multi-pass SDPA decomposition (one attention call per prefix segment).
#
# Both forms collapse to something much simpler. For a query row q:
#   * text token            -> sees kv in [0, q]                       (causal)
#   * image-block token     -> sees kv in [0, q] U [block_start, end)
#                            = [0, end)   because block_start <= q
#   * decode-mode target    -> sees every kv
# In every case the allowed set is a *prefix interval* [0, L(q)]. So the whole
# block-causal structure reduces to a per-row causal-length array, and one
# online-softmax flash kernel with an extra `kv <= L[row]` predicate reproduces
# the mask exactly - no BlockMask, no compilation, no segment loop.
#
# The same kernel also fuses the *prefix KV cache read*: decode steps (denoising
# steps 2..40) attend over [cached prefix K/V, fresh target K/V]. The reference
# torch.cats 32 layers' worth of cache every step; here the kernel reads the two
# sources directly, removing that copy entirely.
#
# Layouts: everything is [B, S, H, D] (the diffusers-native layout); loads
# coalesce along D (stride 1), which is all tl.dot needs.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qwen21_attn_fwd(
    q_ptr,          # [B, SQ, H, D]
    k_cache_ptr,    # [B, P,  H, D]  (prefix K; unused when HAS_CACHE=False)
    v_cache_ptr,    # [B, P,  H, D]
    k_new_ptr,      # [B, SN, H, D]
    v_new_ptr,      # [B, SN, H, D]
    out_ptr,        # [B, SQ, H, D]
    L_ptr,          # [SQ] int32: row attends to kv iff 0 <= kv <= L[row]; -1 -> nothing
    key_valid_ptr,  # [B, KV] uint8 (dummy when HAS_KEY_VALID=False)
    v_scale_ptr,    # [B, H] fp32 per-head V scales (dummy when V_CACHE_FP8=False)
    sm_scale,
    SQ, SN, KV_TOTAL,
    H,              # number of heads (for (b, h) recovery from fused program id)
    P,              # prefix length (0 when HAS_CACHE=False)
    stride_qb, stride_qs, stride_qh,
    stride_kcb, stride_kcs, stride_kch,
    stride_vcb, stride_vcs, stride_vch,
    stride_knb, stride_kns, stride_knh,
    stride_vnb, stride_vns, stride_vnh,
    stride_ob, stride_os, stride_oh,
    stride_valid_b,
    HAS_CACHE: tl.constexpr,
    HAS_KEY_VALID: tl.constexpr,
    V_CACHE_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)  # fused: b * H + h
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_live = offs_m < SQ

    q_ptrs = q_ptr + b * stride_qb + offs_m[:, None] * stride_qs + h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_live[:, None], other=0.0)

    # Per-row prefix limits for this tile
    L = tl.load(L_ptr + offs_m, mask=row_live, other=-1)
    L_max = tl.max(L, axis=0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi_tile = tl.cdiv(L_max + 1, BLOCK_N)

    for phase in tl.static_range(2 if HAS_CACHE else 1):
        if HAS_CACHE and phase == 0:
            seg_len = P
            k_base = k_cache_ptr
            v_base = v_cache_ptr
            sb, ss, sh = stride_kcb, stride_kcs, stride_kch
            vb_, vs_, vh_ = stride_vcb, stride_vcs, stride_vch
            seg_start = 0
            first_tile = 0
        else:
            # phase 1 with cache (fresh target segment), or the single phase
            # when there is no cache (fresh keys cover the whole sequence)
            seg_len = SN
            k_base = k_new_ptr
            v_base = v_new_ptr
            sb, ss, sh = stride_knb, stride_kns, stride_knh
            vb_, vs_, vh_ = stride_vnb, stride_vns, stride_vnh
            seg_start = P
            first_tile = tl.cdiv(P, BLOCK_N) if HAS_CACHE else 0

        tiles_this = tl.cdiv(seg_len, BLOCK_N)
        # tiles of this segment at or below the global hi_tile do work; the
        # rest are beyond every row's prefix limit (Triton has no `break`)
        run_tiles = tl.maximum(0, tl.minimum(tiles_this, hi_tile - first_tile + 1))
        for t in range(0, run_tiles):
            tile_global = first_tile + t

            offs_n_local = t * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_n_abs = seg_start + offs_n_local
            in_seg = offs_n_local < seg_len

            allowed = in_seg[None, :] & (offs_n_abs[None, :] <= L[:, None])

            if HAS_KEY_VALID:
                kv_valid = tl.load(
                    key_valid_ptr + b * stride_valid_b + offs_n_abs, mask=in_seg, other=0
                )
                allowed = allowed & (kv_valid[None, :] != 0)

            k_ptrs = k_base + b * sb + offs_n_local[:, None] * ss + h * sh + offs_d[None, :]
            v_ptrs = v_base + b * vb_ + offs_n_local[:, None] * vs_ + h * vh_ + offs_d[None, :]

            k = tl.load(k_ptrs, mask=in_seg[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N] fp32
            s = tl.where(allowed, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            p = tl.where(allowed, p, 0.0)
            alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_safe))
            l_new = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            if V_CACHE_FP8 and phase == 0:
                # fp8e4m3 cache V with per-head dequant scale; the fresh V in
                # phase 1 is bf16 and takes the plain path below
                vsc = tl.load(v_scale_ptr + b * H + h)
                v = tl.load(v_ptrs, mask=in_seg[:, None], other=0.0).to(tl.float32)
                v = (v * vsc).to(tl.bfloat16)
            else:
                v = tl.load(v_ptrs, mask=in_seg[:, None], other=0.0).to(v_new_ptr.dtype.element_ty)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v_new_ptr.dtype.element_ty), v, acc)
            l_i = l_new

    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    o_ptrs = out_ptr + b * stride_ob + offs_m[:, None] * stride_os + h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_live[:, None])


def block_causal_flash_attention(
    query: torch.Tensor,             # [B, SQ, H, D]
    key: torch.Tensor,               # [B, SN, H, D]  fresh keys
    value: torch.Tensor,             # [B, SN, H, D]
    prefix_lengths: torch.Tensor,    # [SQ] int32
    key_valid: torch.Tensor | None,  # [B, KV_TOTAL] bool or None
    cache_k: torch.Tensor | None = None,  # [B, P, H, D]
    cache_v: torch.Tensor | None = None,
    cache_v_scale: torch.Tensor | None = None,  # [B, H] fp32, only with fp8 cache V
) -> torch.Tensor:
    """Exact block-causal attention via per-row prefix lengths.

    When `cache_k`/`cache_v` are given, keys/values are the *virtual* concat
    [cache, key]; `key`/`value` then hold only the fresh segment and
    `prefix_lengths`/`key_valid` are indexed in absolute (cache-aware) space.
    Rows with `L = -1` attend to nothing and produce zeros. When
    `cache_v_scale` is given the cache V is fp8e4m3 dequantized per head.
    """
    assert query.dtype in (torch.bfloat16, torch.float16), "fp16/bf16 only"
    B, SQ, H, D = query.shape
    if cache_k is not None:
        P = cache_k.shape[1]
        SN = key.shape[1]
        assert cache_k.shape[2] == H and key.shape[2] == H
        assert cache_k.dtype == key.dtype
    else:
        P = 0
        SN = key.shape[1]

    KV_TOTAL = P + SN
    out = torch.empty_like(query)

    if key_valid is not None:
        kv_bytes = key_valid.to(torch.uint8).contiguous()
        has_valid = True
        stride_valid_b = kv_bytes.stride(0)
    else:
        kv_bytes = query  # dummy pointer, never loaded
        has_valid = False
        stride_valid_b = 0

    v_fp8 = cache_v_scale is not None
    if v_fp8:
        assert cache_v is not None and cache_v.dtype == torch.float8_e4m3fn
        v_scale_arg = cache_v_scale
    else:
        v_scale_arg = query  # dummy pointer, never loaded

    # Shape-dependent configs (swept on sm_120), all within the 101 KB shared
    # memory limit. The fp8 cache-V variant allocates conversion buffers, so it
    # runs one fewer stage.
    if v_fp8:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 8, 2
    elif KV_TOTAL <= 2048:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 128, 8, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 8, 3

    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _qwen21_attn_fwd[grid](
        query,
        cache_k if P > 0 else query, cache_v if P > 0 else query,
        key, value, out,
        prefix_lengths, kv_bytes, v_scale_arg,
        query.shape[-1] ** -0.5,
        SQ, SN, KV_TOTAL,
        H, P,
        query.stride(0), query.stride(1), query.stride(2),
        cache_k.stride(0) if P > 0 else 0, cache_k.stride(1) if P > 0 else 0, cache_k.stride(2) if P > 0 else 0,
        cache_v.stride(0) if P > 0 else 0, cache_v.stride(1) if P > 0 else 0, cache_v.stride(2) if P > 0 else 0,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        stride_valid_b,
        HAS_CACHE=P > 0,
        HAS_KEY_VALID=has_valid,
        V_CACHE_FP8=v_fp8,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Reference (PyTorch) implementation of the same semantics, for unit tests.
# ---------------------------------------------------------------------------
def reference_block_causal_attention(
    query: torch.Tensor,             # [B, SQ, H, D]
    key: torch.Tensor,               # [B, SN, H, D]  (fresh)
    value: torch.Tensor,
    prefix_lengths: torch.Tensor,    # [SQ]
    key_valid: torch.Tensor | None,
    cache_k: torch.Tensor | None = None,
    cache_v: torch.Tensor | None = None,
) -> torch.Tensor:
    B, SQ, H, D = query.shape
    P = 0 if cache_k is None else cache_k.shape[1]
    k_all = key if cache_k is None else torch.cat([cache_k, key], dim=1)
    v_all = value if cache_v is None else torch.cat([cache_v, value], dim=1)
    KV = k_all.shape[1]

    idx = torch.arange(KV, device=query.device)
    allowed = idx[None, None, None, :] <= prefix_lengths[None, None, :, None]  # [B?, SQ, KV]
    allowed = allowed.expand(B, -1, -1, -1) if allowed.shape[0] == 1 else allowed
    if key_valid is not None:
        allowed = allowed & key_valid[:, None, None, :].to(torch.bool)

    q = query.transpose(1, 2).float()   # [B, H, SQ, D]
    k = k_all.transpose(1, 2).float()
    v = v_all.transpose(1, 2).float()
    scores = q @ k.transpose(-1, -2) * (D ** -0.5)
    scores = scores.masked_fill(~allowed, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn.to(v.dtype) @ v          # [B, H, SQ, D]
    return out.transpose(1, 2).to(query.dtype)