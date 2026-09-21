# Probe: Blackwell (sm_120) FP8 capabilities for the kernel runtime.
import torch

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("torch:", torch.__version__)

M, K, N = 256, 512, 512
a = torch.randn(M, K, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
b = torch.randn(N, K, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)

# tensorwise
try:
    sa = torch.ones(1, device="cuda")
    sb = torch.ones(1, device="cuda")
    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    print("scaled_mm tensorwise: OK", tuple(out.shape), out.dtype)
except Exception as e:
    print("scaled_mm tensorwise: FAIL ->", e)

# rowwise
try:
    sa = torch.ones(M, 1, device="cuda")
    sb = torch.ones(1, N, device="cuda")
    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    print("scaled_mm rowwise:    OK", tuple(out.shape), out.dtype)
except Exception as e:
    print("scaled_mm rowwise:    FAIL ->", e)

# bf16 matmul sanity
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul:", (x @ w).shape, "OK")

# quick fp8 vs bf16 relative error on a random linear
g = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.5
wgt = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
ref = (g.float() @ wgt.float().t())
s_w = wgt.abs().amax(dim=1) / 448.0
wq = (wgt / s_w[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
s_a = g.abs().amax(dim=1, keepdim=True) / 448.0
aq = (g / s_a).clamp(-448, 448).to(torch.float8_e4m3fn)
out8 = torch._scaled_mm(aq, wq.t(), scale_a=s_a, scale_b=s_w.t()[None] if False else s_w[:, None].t().reshape(1, -1) if False else s_w.reshape(1, -1), out_dtype=torch.bfloat16)
# scale_b must be [1, N]; s_w is [N] -> reshape
rel = (out8.float() - ref).norm() / ref.norm()
print(f"fp8 rowwise rel-error vs fp32: {rel.item():.5f}")