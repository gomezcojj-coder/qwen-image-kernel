# Qwen-Image-2.1 optimization kernels for NVIDIA Blackwell consumer GPUs (Windows).
#
# This package implements:
#   * a single fused Triton flash-attention kernel that expresses Qwen-Image 2.1's
#     block-causal mask exactly via per-row prefix lengths (no torch.compile / flex_attention
#     required, which is what makes the model painful to run on Windows),
#   * fused RMSNorm+RoPE / LayerNorm+modulation / SwiGLU kernels,
#   * an FP8 (W8A8, dynamic per-token activations) linear path built on torch._scaled_mm,
#   * a drop-in attention processor and a 12 GB-friendly pipeline runtime.

from .processors import QwenImage21KernelProcessor  # noqa: F401
from .runtime import KernelRuntime  # noqa: F401

__version__ = "0.1.0"