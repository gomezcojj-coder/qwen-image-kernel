# Probe 3: FP8 scaling modes with proper fp32 scales.
import torch

print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.manual_seed(0)

M, K, N = 1024, 4096, 4096
g = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
wgt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
ref = g.float() @ wgt.float().t()

def relerr(out):
    return ((out.float() - ref).norm() / ref.norm()).item()

# 1) tensorwise
try:
    aq = (g / g.abs().amax()).clamp(-448, 448).to(torch.float8_e4m3fn)
    wq = (wgt / wgt.abs().amax()).clamp(-448, 448).to(torch.float8_e4m3fn)
    sa = (g.abs().amax() / 448.0).float().reshape(1)
    sb = (wgt.abs().amax() / 448.0).float().reshape(1)
    out = torch._scaled_mm(aq, wq.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    print(f"tensorwise      : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print("tensorwise      : FAIL ->", str(e)[:150])

# 2) blockwise 1x128 x 128x128
try:
    KB = K // 128
    NB = N // 128
    a_blk = g.float().reshape(M, KB, 128)
    s_a = (a_blk.abs().amax(dim=-1) / 448.0).float().clamp_min(1e-12)     # [M, KB]
    aq = (a_blk / s_a[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
    w_blk = wgt.float().reshape(NB, 128, KB, 128)
    s_w = (w_blk.abs().amax(dim=(1, 3)) / 448.0).float().clamp_min(1e-12) # [NB, KB]
    wq = (w_blk / s_w[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)
    scale_a = s_a.t().contiguous().t()   # [M, KB], outer-dim-major
    scale_b = s_w.t().contiguous()       # [KB, NB], contiguous
    out = torch._scaled_mm(aq, wq.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    print(f"blockwise 1x128 : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print(f"blockwise 1x128 : FAIL -> {str(e)[:300]}")

# 3) MXFP8 1x32 e8m0
try:
    KB32 = K // 32
    NB32 = N // 32
    a_blk = g.float().reshape(M, KB32, 32)
    amax_a = a_blk.abs().amax(dim=-1)                                     # [M, KB32]
    exp_a = torch.ceil(torch.log2((amax_a / 448.0).clamp_min(2.0 ** -127)))
    s_a_e8m0 = torch.exp2(exp_a).to(torch.float8_e8m0fnu)                 # [M, KB32]
    aq = (a_blk / torch.exp2(exp_a)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)

    w_flat = wgt.float().reshape(NB32, 32)  # wrong grouping: need [N, KB32] per-row blocks
    w_blk = wgt.float().reshape(N, KB32, 32)
    amax_w = w_blk.abs().amax(dim=-1)                                     # [N, KB32]
    exp_w = torch.ceil(torch.log2((amax_w / 448.0).clamp_min(2.0 ** -127)))
    s_w_e8m0 = torch.exp2(exp_w).to(torch.float8_e8m0fnu)
    wq = (w_blk / torch.exp2(exp_w)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)

    # try both scale layouts
    for name, (sca, scb) in {
        "as-is [M,KB32],[N,KB32]": (s_a_e8m0, s_w_e8m0),
        "a.t-contig-t, b.t-contig": (s_a_e8m0.t().contiguous().t(), s_w_e8m0.t().contiguous()),
    }.items():
        try:
            out = torch._scaled_mm(aq, wq.t(), scale_a=sca, scale_b=scb, out_dtype=torch.bfloat16)
            print(f"mxfp8 1x32 {name}: OK  rel-err {relerr(out):.5f}")
            break
        except Exception as e:
            print(f"mxfp8 1x32 {name}: FAIL -> {str(e)[:160]}")
except Exception as e:
    print(f"mxfp8 1x32      : FAIL -> {str(e)[:160]}")