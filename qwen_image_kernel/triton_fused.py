# Fused elementwise/norm kernels for the Qwen-Image 2.1 kernel runtime.
#
# All kernels replicate the stock diffusers ops' numeric semantics (fp32
# compute, bf16 rounding at the same points) so the kernel path tracks the
# BF16 reference closely.
#
#  * rms_rope_kernel       - per-head RMSNorm (diffusers convention: plain
#                            fp32 weight multiply) + complex RoPE, fused,
#                            including the stock bf16 round-trip between norm
#                            and rotation for numerical parity.
#  * ln_modulate_kernel    - LayerNorm (no affine) * (1 + scale), with a
#                            per-token modulation row map (target rows -> own
#                            timestep row; prefix rows -> t = 0 row).
#  * gate_residual_kernel  - x + tanh(gate) * f, same row selection.
#  * swiglu_kernel         - silu(gate_proj) * up_proj in one pass.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# Fused per-head RMSNorm + RoPE
# ---------------------------------------------------------------------------
@triton.jit
def rms_rope_kernel(
    x_ptr,    # [B*S*H, D] bf16, D contiguous
    out_ptr,
    w_ptr,    # [D] fp32 RMSNorm weight
    cos_ptr,  # [S, HALF] fp32
    sin_ptr,  # [S, HALF] fp32
    NROWS,    # B*S*H
    S, H,
    eps,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    live = offs_r < NROWS
    offs_h = tl.arange(0, HALF)
    base = offs_r[:, None] * D

    x_even = tl.load(x_ptr + base + (2 * offs_h)[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + base + (2 * offs_h + 1)[None, :], mask=live[:, None], other=0.0).to(tl.float32)

    ss = tl.sum(x_even * x_even + x_odd * x_odd, axis=1)
    rrms = tl.rsqrt(ss / D + eps)

    w_e = tl.load(w_ptr + 2 * offs_h)
    w_o = tl.load(w_ptr + 2 * offs_h + 1)
    # stock: fp32 norm, single rounding to bf16, back to fp32 for rotation
    n_e = ((x_even * rrms[:, None]) * w_e[None, :]).to(tl.bfloat16).to(tl.float32)
    n_o = ((x_odd * rrms[:, None]) * w_o[None, :]).to(tl.bfloat16).to(tl.float32)

    s_idx = (offs_r // H) % S
    cos = tl.load(cos_ptr + s_idx[:, None] * HALF + offs_h[None, :], mask=live[:, None], other=0.0)
    sin = tl.load(sin_ptr + s_idx[:, None] * HALF + offs_h[None, :], mask=live[:, None], other=0.0)

    # (e + i*o) * (c + i*s) = (e*c - o*s) + i*(e*s + o*c)
    o_even = n_e * cos - n_o * sin
    o_odd = n_e * sin + n_o * cos

    tl.store(out_ptr + base + (2 * offs_h)[None, :], o_even.to(tl.bfloat16), mask=live[:, None])
    tl.store(out_ptr + base + (2 * offs_h + 1)[None, :], o_odd.to(tl.bfloat16), mask=live[:, None])


def fused_rms_norm_rope(
    x: torch.Tensor,           # [B, S, H, D] bf16
    weight: torch.Tensor,      # [D] fp32
    freqs_cis: torch.Tensor,   # complex [S, D/2]
    eps: float,
) -> torch.Tensor:
    B, S, H, D = x.shape
    cos = freqs_cis.real.float().contiguous()
    sin = freqs_cis.imag.float().contiguous()
    out = torch.empty_like(x)
    nrows = B * S * H
    BLOCK_R = 16
    rms_rope_kernel[(triton.cdiv(nrows, BLOCK_R),)](
        x, out, weight, cos, sin,
        nrows, S, H, eps,
        D=D, HALF=D // 2, BLOCK_R=BLOCK_R,
        num_warps=2, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# LayerNorm + (1 + scale) with modulation row map
# ---------------------------------------------------------------------------
@triton.jit
def ln_modulate_kernel(
    x_ptr,        # [B*S, D] bf16
    mod_ptr,      # modulation rows, bf16
    rowmap_ptr,   # [S] int32: 0 for real-timestep rows, B for the t=0 row
    out_ptr,
    S,
    D: tl.constexpr,
    SCALE_OFF, MOD_STRIDE,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    s = row % S
    b = row // S
    mod_row = tl.load(rowmap_ptr + s) + b
    offs_d = tl.arange(0, BLOCK_D)
    live = offs_d < D

    x = tl.load(x_ptr + row * D + offs_d, mask=live, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(live, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    y = (xc * tl.rsqrt(var + eps)).to(tl.bfloat16)  # torch LayerNorm rounding point
    scale = tl.load(mod_ptr + mod_row * MOD_STRIDE + SCALE_OFF + offs_d, mask=live, other=0.0)
    y = y * (1.0 + scale)
    tl.store(out_ptr + row * D + offs_d, y, mask=live)


def _modulation_row_map(
    target_token_mask: torch.Tensor | None, B: int, S: int, device: torch.device
) -> torch.Tensor:
    """Per-position modulation row: 0 = own sample's timestep row, B = t=0 row."""
    if target_token_mask is not None:
        return torch.where(
            target_token_mask.to(torch.bool),
            torch.zeros(S, dtype=torch.int32, device=device),
            torch.full((S,), B, dtype=torch.int32, device=device),
        )
    return torch.zeros(S, dtype=torch.int32, device=device)


def fused_layer_norm_modulate(
    x: torch.Tensor,              # [B, S, D] bf16
    modulation: torch.Tensor,     # scale source; row-major [R, >=scale_offset+D]
    scale_offset: int,
    modulation_stride: int,
    target_token_mask: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """LayerNorm then (1 + scale). `modulation` holds one row per sample plus a
    trailing t=0 row under causal_condition (rows [0,B) real, row B = t=0)."""
    B, S, D = x.shape
    out = torch.empty_like(x)
    row_map = _modulation_row_map(target_token_mask, B, S, x.device)
    ln_modulate_kernel[(B * S,)](
        x, modulation, row_map, out,
        S,
        D=D,
        SCALE_OFF=scale_offset,
        MOD_STRIDE=modulation_stride,
        eps=eps,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Gate + residual:  y = x + tanh(gate_sel) * f
# ---------------------------------------------------------------------------
@triton.jit
def gate_residual_kernel(
    x_ptr,       # [B*S, D] bf16
    f_ptr,       # [B*S, D] bf16
    mod_ptr,     # modulation rows, bf16
    rowmap_ptr,  # [S] int32
    out_ptr,
    S,
    D: tl.constexpr,
    GATE_OFF, MOD_STRIDE,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    s = row % S
    b = row // S
    mod_row = tl.load(rowmap_ptr + s) + b
    offs_d = tl.arange(0, BLOCK_D)
    live = offs_d < D

    x = tl.load(x_ptr + row * D + offs_d, mask=live, other=0.0)
    f = tl.load(f_ptr + row * D + offs_d, mask=live, other=0.0)
    g = tl.load(mod_ptr + mod_row * MOD_STRIDE + GATE_OFF + offs_d, mask=live, other=0.0)
    g = libdevice.tanh(g.to(tl.float32)).to(tl.bfloat16)  # stock tanh rounding point

    y = x + g * f
    tl.store(out_ptr + row * D + offs_d, y, mask=live)


def fused_gate_residual(
    x: torch.Tensor,              # [B, S, D]
    f: torch.Tensor,              # [B, S, D]  (attention or mlp output)
    modulation: torch.Tensor,     # [B(+1), 4*inner]
    gate_offset: int,
    target_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    B, S, D = x.shape
    out = torch.empty_like(x)
    row_map = _modulation_row_map(target_token_mask, B, S, x.device)
    gate_residual_kernel[(B * S,)](
        x, f, modulation, row_map, out,
        S,
        D=D,
        GATE_OFF=gate_offset,
        MOD_STRIDE=modulation.stride(0),
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# SwiGLU:  y = silu(gate) * up
# ---------------------------------------------------------------------------
@triton.jit
def swiglu_kernel(
    g_ptr, p_ptr, out_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    g = tl.load(g_ptr + offs, mask=live, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + offs, mask=live, other=0.0)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16)  # stock SiLU rounding point
    y = s * p
    tl.store(out_ptr + offs, y, mask=live)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gate, up: [B, S, M] bf16 -> silu(gate) * up."""
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 4096
    swiglu_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK, num_warps=4)
    return out