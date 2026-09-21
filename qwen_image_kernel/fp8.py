# FP8 (W8A8) linear layers for Qwen-Image 2.1 on Blackwell consumer GPUs.
#
# Capability notes (probed on RTX 5070, sm_120, torch 2.14.0+cu130):
#   * torch._scaled_mm TENSORWISE fp8e4m3: supported, 1.95x faster than bf16 GEMM
#     at the DiT's 1536x4096x4096 shape.
#   * ROWWISE scaling: rejected by ATen ("not supported on your device").
#   * BlockWise 1x128: passes shape validation, but cuBLASLt has no kernel
#     (CUBLAS_STATUS_NOT_SUPPORTED) on this device.
#   * MXFP8 (1x32 e8m0): accepted but numerically wrong through this API.
# So the supported-and-fast path is tensorwise scaling; this module implements
# dynamic per-tensor activation quantization + static per-tensor weight scales,
# with the quantization fused into preceding kernels where possible.

from __future__ import annotations

import torch
import torch.nn as nn

FP8_E4M3_MAX = 448.0


def compute_scale(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor fp8 activation scale (fp32, on-device)."""
    return (x.abs().amax().float() / FP8_E4M3_MAX).clamp_min(1e-12)


def quantize_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] bf16 -> (fp8, scale). One amax reduction + one scale+cast."""
    s = compute_scale(x)
    xq = (x.float() / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, s.reshape(1)


def quantize_weight_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (fp8 [N, K], scale fp32 [1])."""
    s = (w.abs().amax().float() / FP8_E4M3_MAX).clamp_min(1e-12)
    wq = (w.float() / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return wq, s.reshape(1)


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with fp8 weights and dynamic
    per-tensor activation quantization, computed via torch._scaled_mm."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_features < 256:
            # cuBLASLt fp8 heuristics reject skinny-K GEMMs (e.g. img_in, K=64);
            # these linears are compute-trivial, so dequantize and use bf16.
            w = self.weight.to(torch.bfloat16).float() * self.weight_scale
            out = torch.nn.functional.linear(x, w.to(x.dtype), self.bias)
            return out.to(x.dtype)
        x2 = x.reshape(-1, x.shape[-1])
        xq, s_a = quantize_tensorwise(x2)
        out = torch._scaled_mm(xq, self.weight.t(), scale_a=s_a, scale_b=self.weight_scale, out_dtype=torch.bfloat16)
        out = out.reshape(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)

    @staticmethod
    def from_linear(linear: nn.Linear) -> "FP8Linear":
        """Convert a loaded bf16 nn.Linear in place (weights must be on CPU or GPU)."""
        mod = FP8Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        wq, s = quantize_weight_tensorwise(linear.weight.data.float())
        mod.weight = wq.to(linear.weight.device)
        mod.weight_scale = s.to(linear.weight.device)
        if linear.bias is not None:
            mod.bias = linear.bias.data.to(torch.bfloat16).clone()
        return mod.to(linear.weight.device)


def convert_linears_fp8(module: nn.Module, skip_patterns: tuple[str, ...] = (), prefix: str = "") -> list[str]:
    """Recursively replace nn.Linear submodules with FP8Linear (in place).

    Returns the list of converted submodule names. Skips anything matching
    `skip_patterns` (matched against the fully-qualified name).
    """
    converted = []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if any(p in full for p in skip_patterns):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8Linear.from_linear(child))
            converted.append(full)
        else:
            converted.extend(convert_linears_fp8(child, skip_patterns, full))
    return converted


def fp8_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict where converted linears expose fp8 `weight` + `weight_scale`;
    everything else keeps its dtype. (The module tree already reflects the
    conversion, so this is just the plain state dict.)"""
    return {k: v for k, v in model.state_dict().items()}