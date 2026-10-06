# One-time FP8 cache builder: quantizes the DiT and text-encoder linears from
# the BF16 checkpoint and writes mixed fp8/bf16 safetensors that the runtime
# loads into a meta-device skeleton (params only; non-persistent buffers are
# created live at init).

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file

from .fp8 import FP8Linear, quantize_weight_tensorwise
from .processors import QwenImage21KernelBlock

_QUANT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _staged_quantize(w: torch.Tensor):
    """Quantize on GPU when the source lives on CPU (the amax reduction is far
    faster there; weights are moved one linear at a time)."""
    if w.device.type == "cpu" and _QUANT_DEVICE == "cuda":
        w_gpu = w.to(_QUANT_DEVICE, non_blocking=True)
        wq, s = quantize_weight_tensorwise(w_gpu)
        return wq.to("cpu"), s.to("cpu")
    return quantize_weight_tensorwise(w)


def _convert_module_fp8(module: torch.nn.Module, skip: tuple[str, ...] = ()) -> list[str]:
    converted = []
    for parent_name, parent_mod in list(module.named_modules()):
        for child_name, child in list(parent_mod.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(p in full for p in skip):
                continue
            if isinstance(child, torch.nn.Linear):
                in_f, out_f, has_b = child.in_features, child.out_features, child.bias is not None
                w = child.weight.data
                bias = child.bias.data if has_b else None
                wq, s = _staged_quantize(w)
                new = FP8Linear(in_f, out_f, bias=has_b)
                new.weight = wq
                new.weight_scale = s
                if has_b:
                    new.bias = bias.to(torch.bfloat16).clone()
                setattr(parent_mod, child_name, new)
                converted.append(full)
    return converted


def _sd_for_save(model: torch.nn.Module, drop_patterns: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """State dict safe for safetensors.save_file (splits shared storage)."""
    sd: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for name, t in model.state_dict().items():
        if any(p in name for p in drop_patterns):
            continue
        ptr = t.untyped_storage().data_ptr()
        if t._base is not None or ptr in seen:
            t = t.clone()
            ptr = t.untyped_storage().data_ptr()
        seen.add(ptr)
        sd[name] = t.contiguous()
    return sd


def _fuse_block_linears(block) -> None:
    """Replace a kernel block's to_q/to_k/to_v and img_mlp gate_layer/proj with
    single fused FP8 linears, quantized once from the BF16 weights (same
    one-rounding error class as the individual quantization - no accuracy
    cost), and drop the individual linears (VRAM-neutral: the fused weight is
    the concatenation)."""
    attn, mlp = block.attn, block.img_mlp
    q, k, v = attn.to_q, attn.to_k, attn.to_v
    assert q.in_features == k.in_features == v.in_features
    assert q.out_features == k.out_features == v.out_features
    assert (q.bias is not None) == (k.bias is not None) == (v.bias is not None)
    inner = q.out_features
    fused_qkv = FP8Linear(q.in_features, 3 * inner, bias=q.bias is not None)
    w = torch.cat([q.weight.data, k.weight.data, v.weight.data], 0)
    wq, s = _staged_quantize(w)
    fused_qkv.weight = wq
    fused_qkv.weight_scale = s
    if q.bias is not None:
        fused_qkv.bias = torch.cat([q.bias.data, k.bias.data, v.bias.data], 0).to(torch.bfloat16)
    attn.fused_qkv = fused_qkv.to(w.device)
    attn.to_q = torch.nn.Identity()
    attn.to_k = torch.nn.Identity()
    attn.to_v = torch.nn.Identity()

    gate, proj = mlp.gate_layer, mlp.proj
    assert gate.in_features == proj.in_features and gate.out_features == proj.out_features
    assert (gate.bias is not None) == (proj.bias is not None)
    hidden = gate.out_features
    fused_up = FP8Linear(gate.in_features, 2 * hidden, bias=gate.bias is not None)
    w = torch.cat([gate.weight.data, proj.weight.data], 0)
    wq, s = _staged_quantize(w)
    fused_up.weight = wq
    fused_up.weight_scale = s
    if gate.bias is not None:
        fused_up.bias = torch.cat([gate.bias.data, proj.bias.data], 0).to(torch.bfloat16)
    mlp.fused_up = fused_up.to(w.device)
    mlp.gate_layer = torch.nn.Identity()
    mlp.proj = torch.nn.Identity()


def build_transformer_fp8_cache(model_id: str, out_path: str, fused_linears: bool = True) -> None:
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[fp8-cache] loading BF16 transformer from {model_id} ...", flush=True)
    model = QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # kernel block shells keep submodule names identical, so keys map 1:1
    model.transformer_blocks = torch.nn.ModuleList(
        [QwenImage21KernelBlock(blk) for blk in model.transformer_blocks]
    )

    if fused_linears:
        print("[fp8-cache] fusing qkv / mlp-up linears (single-rounding) ...", flush=True)
        for blk in model.transformer_blocks:
            _fuse_block_linears(blk)

    print("[fp8-cache] quantizing block linears to fp8 ...", flush=True)
    _convert_module_fp8(model)

    # the non-persistent timestep freqs buffer is rebuilt at init - drop it
    sd = _sd_for_save(model, drop_patterns=(".freqs",))
    save_file(sd, out_path)
    print(f"[fp8-cache] wrote {out_path} ({len(sd)} tensors)", flush=True)
    del model
    torch.cuda.empty_cache()


def build_text_encoder_fp8_cache(model_id: str, out_path: str) -> None:
    from transformers import Qwen3VLForConditionalGeneration

    print(f"[fp8-cache] loading BF16 text encoder from {model_id} ...", flush=True)
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    print("[fp8-cache] quantizing text-encoder linears to fp8 ...", flush=True)
    _convert_module_fp8(te)

    sd = _sd_for_save(te)
    save_file(sd, out_path)
    print(f"[fp8-cache] wrote {out_path} ({len(sd)} tensors)", flush=True)
    del te
    torch.cuda.empty_cache()