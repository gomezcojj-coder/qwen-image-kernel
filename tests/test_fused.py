# Unit tests: fused Triton kernels vs plain-torch stock semantics.
import sys

import torch

from qwen_image_kernel.triton_fused import (
    fused_gate_residual,
    fused_layer_norm_modulate,
    fused_rms_norm_rope,
    fused_swiglu,
)

torch.manual_seed(0)
DEV = "cuda"


def torch_rms_norm_rope(x, weight, freqs_cis, eps):
    # stock diffusers semantics
    import torch.nn.functional as F

    var = x.float().pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)          # fp32 out (weight fp32)
    y = y * weight.float()
    y = y.to(x.dtype)                        # .to(value.dtype)
    # complex rope
    xc = torch.view_as_complex(y.float().reshape(*y.shape[:-1], -1, 2))
    rot = xc * freqs_cis.unsqueeze(1)  # [S, 1, D/2] broadcast over B, H
    out = torch.view_as_real(rot).flatten(-2)
    return out.to(x.dtype)


def test_rms_rope():
    B, S, H, D = 1, 300, 32, 128
    x = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16) * 0.7
    w = torch.randn(D, device=DEV, dtype=torch.float32) * 0.3
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=DEV).float() / D))
    ang = torch.outer(torch.arange(S, device=DEV).float(), inv)
    freqs = torch.polar(torch.ones_like(ang), ang)  # [S, D/2] complex
    mine = fused_rms_norm_rope(x, w, freqs, 1e-6)
    ref = torch_rms_norm_rope(x, w, freqs, 1e-6)
    d = (mine.float() - ref.float()).abs().max().item()
    print(f"rms+rope            max|diff| {d:.6f} -> {'PASS' if d < 4e-2 else 'FAIL'}")
    return d < 4e-2


def torch_ln_modulate(x, modulation, scale_off, mod_stride, row_map, eps):
    B, S, D = x.shape
    x32 = x.float()
    mu = x32.mean(-1, keepdim=True)
    var = x32.var(-1, unbiased=False, keepdim=True)
    y = ((x32 - mu) / torch.sqrt(var + eps)).to(torch.bfloat16)
    rows = row_map.long()  # [S]
    mod = modulation  # [R, >= scale_off + D]
    sel = mod[rows][:, scale_off : scale_off + D]  # [S, D]
    y = y * (1 + sel)
    return y


def test_ln_modulate():
    B, S, D = 1, 500, 4096
    x = torch.randn(B, S, D, device=DEV, dtype=torch.bfloat16)
    R = B + 1
    mod = torch.randn(R, 4 * D, device=DEV, dtype=torch.bfloat16) * 0.2
    tmask = torch.rand(S, device=DEV) > 0.5
    row_map = torch.where(tmask, torch.zeros(S, dtype=torch.int32, device=DEV),
                          torch.full((S,), B, dtype=torch.int32, device=DEV))
    mine = fused_layer_norm_modulate(x, mod, 0, 4 * D, tmask, 1e-6)
    ref = torch_ln_modulate(x, mod, 0, 4 * D, row_map, 1e-6)
    d = (mine.float() - ref.float()).abs().max().item()
    print(f"ln+modulate         max|diff| {d:.6f} -> {'PASS' if d < 4e-2 else 'FAIL'}")
    return d < 4e-2


def test_gate_residual():
    B, S, D = 1, 500, 4096
    x = torch.randn(B, S, D, device=DEV, dtype=torch.bfloat16)
    f = torch.randn(B, S, D, device=DEV, dtype=torch.bfloat16) * 2
    R = B + 1
    mod = torch.randn(R, 4 * D, device=DEV, dtype=torch.bfloat16) * 0.2
    tmask = torch.rand(S, device=DEV) > 0.5
    row_map = torch.where(tmask, torch.zeros(S, dtype=torch.int32, device=DEV),
                          torch.full((S,), B, dtype=torch.int32, device=DEV))
    mine = fused_gate_residual(x, f, mod, D, tmask)
    sel = mod[row_map.long()][:, D : 2 * D]  # gate_offset = D (mod1.gate)
    ref = x + torch.tanh(sel) * f
    d = (mine.float() - ref.float()).abs().max().item()
    print(f"gate+residual       max|diff| {d:.6f} -> {'PASS' if d < 4e-2 else 'FAIL'}")
    return d < 4e-2


def test_swiglu():
    B, S, M = 1, 300, 12288
    g = torch.randn(B, S, M, device=DEV, dtype=torch.bfloat16)
    p = torch.randn(B, S, M, device=DEV, dtype=torch.bfloat16)
    mine = fused_swiglu(g, p)
    ref = torch.nn.functional.silu(g) * p
    d = (mine.float() - ref.float()).abs().max().item()
    print(f"swiglu              max|diff| {d:.6f} -> {'PASS' if d < 4e-2 else 'FAIL'}")
    return d < 4e-2


ok = test_rms_rope() and test_ln_modulate() and test_gate_residual() and test_swiglu()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)