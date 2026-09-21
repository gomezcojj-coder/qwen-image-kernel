# Probe 5: MXFP8 (1x32 e8m0 scales) - the Blackwell-native FP8 microscaling path.
import torch

print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.manual_seed(0)

M, K, N = 1024, 4096, 4096
g = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
wgt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
ref = g.float() @ wgt.float().t()

def relerr(out):
    return ((out.float() - ref).norm() / ref.norm()).item()

KB32 = K // 32  # 128
a_blk = g.float().reshape(M, KB32, 32)
amax_a = a_blk.abs().amax(dim=-1)                                      # [M, KB32]
exp_a = torch.ceil(torch.log2((amax_a / 448.0).clamp_min(2.0 ** -127)))
s_a8 = torch.exp2(exp_a).to(torch.float8_e8m0fnu)                      # [M, KB32]
aq = (a_blk / torch.exp2(exp_a)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)

w_blk = wgt.float().reshape(N, KB32, 32)
amax_w = w_blk.abs().amax(dim=-1)                                      # [N, KB32]
exp_w = torch.ceil(torch.log2((amax_w / 448.0).clamp_min(2.0 ** -127)))
s_w8 = torch.exp2(exp_w).to(torch.float8_e8m0fnu)                      # [N, KB32]
wq = (w_blk / torch.exp2(exp_w)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)

print("s_a8:", tuple(s_a8.shape), s_a8.dtype, "s_w8:", tuple(s_w8.shape), s_w8.dtype)
attempts = {
    "a[M,KB32]rm + b[N,KB32]rm": (s_a8, s_w8),
    "a[M,KB32]rm + b.t()[KB32,N]c": (s_a8, s_w8.t().contiguous()),
    "a.t().c().t() + b[N,KB32]rm": (s_a8.t().contiguous().t(), s_w8),
}
for name, (sca, scb) in attempts.items():
    try:
        out = torch._scaled_mm(aq, wq.t(), scale_a=sca, scale_b=scb, out_dtype=torch.bfloat16)
        print(f"mxfp8 {name}: OK  rel-err {relerr(out):.5f}")
        break
    except Exception as e:
        print(f"mxfp8 {name}: FAIL -> {str(e).splitlines()[0][:140]}")

# benchmark tensorwise fp8 vs bf16 GEMM at DiT-like shapes, M=1536
def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters

M2 = 1536
g2 = torch.randn(M2, 4096, device="cuda", dtype=torch.bfloat16)
w2 = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
s_a2 = (g2.abs().amax() / 448.0).float().reshape(1)
aq2 = (g2 / s_a2).clamp(-448, 448).to(torch.float8_e4m3fn)
wq2 = (w2 / (w2.abs().amax() / 448.0).float()).clamp(-448, 448).to(torch.float8_e4m3fn)
s_b2 = (w2.abs().amax() / 448.0).float().reshape(1)
t_bf16 = bench(lambda: g2 @ w2.t())
t_fp8 = bench(lambda: torch._scaled_mm(aq2, wq2.t(), scale_a=s_a2, scale_b=s_b2, out_dtype=torch.bfloat16))
print(f"GEMM 1536x4096x4096: bf16 {t_bf16:.3f} ms | fp8 tensorwise {t_fp8:.3f} ms | ratio {t_bf16/t_fp8:.2f}x")