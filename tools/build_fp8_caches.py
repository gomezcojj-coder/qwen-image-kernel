# One-time: build the FP8 weight caches from the BF16 checkpoint.
# Run this once after installing; the runtime loads the caches from fp8_cache/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_image_kernel.quantize_offline import (
    build_text_encoder_fp8_cache,
    build_transformer_fp8_cache,
)

build_transformer_fp8_cache("Qwen/Qwen-Image-2.1", "fp8_cache/transformer_fp8.safetensors")
build_text_encoder_fp8_cache("Qwen/Qwen-Image-2.1", "fp8_cache/text_encoder_fp8.safetensors")