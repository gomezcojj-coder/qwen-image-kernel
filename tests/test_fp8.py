# Unit test: FP8Linear vs nn.Linear (fp32 reference) on realistic magnitudes.
import sys

import torch

from qwen_image_kernel.fp8 import FP8Linear

torch.manual_seed(0)
DEV = "cuda"

ok = True
for (M, K, N, label) in [(1536, 4096, 4096, "qkv-like"), (1536, 12288, 4096, "mlp-out-like")]:
    lin = torch.nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device=DEV)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    ref = (x.float() @ lin.weight.float().t())

    fq = FP8Linear.from_linear(lin).to(DEV)
    out = fq(x)

    rel = ((out.float() - ref).norm() / ref.norm()).item()
    good = rel < 0.06
    ok &= good
    print(f"FP8Linear {label:12s} M={M} K={K} N={N}  rel-err {rel:.4f} -> {'PASS' if good else 'FAIL'}")

# bf16 comparison for context
lin = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16, device=DEV)
with torch.no_grad():
    lin.weight.mul_(0.02)
x = torch.randn(1536, 4096, device=DEV, dtype=torch.bfloat16)
ref = x.float() @ lin.weight.float().t()
out_bf16 = (x @ lin.weight.t()).float()
rel_bf16 = ((out_bf16 - ref).norm() / ref.norm()).item()
print(f"reference bf16 GEMM rel-err {rel_bf16:.4f} (for context)")

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)