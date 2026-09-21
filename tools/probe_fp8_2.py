# Probe 2: which FP8 scaling modes actually work on sm_120, with fp32 scales.
import torch

print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.manual_seed(0)

M, K, N = 1024, 4096, 4096
g = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5)
wgt = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
ref = g.float() @ wgt.float().t()

def relerr(out):
    return ((out.float() - ref).norm() / ref.norm()).item()

# 1) tensorwise, fp32 scales
try:
    aq = (g / g.abs().amax()).clamp(-448, 448).to(torch.float8_e4m3fn)
    wq = (wgt / wgt.abs().amax()).clamp(-448, 448).to(torch.float8_e4m3fn)
    sa = (g.abs().amax() / 448.0).float().reshape(1)
    sb = (wgt.abs().amax() / 448.0).reshape(1)
    out = torch._scaled_mm(aq, wq.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    print(f"tensorwise        : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print("tensorwise        : FAIL ->", str(e)[:120])

# 2) blockwise 1x128 (activations) x 128x128 (weights)  [DeepSeek-style]
try:
    KB = K // 128
    # activation: 1x128 along K
    a_blk = g.float().reshape(M, KB, 128)
    s_a = a_blk.abs().amax(dim=-1, keepdim=False) / 448.0          # [M, KB]
    s_a_safe = s_a.clamp_min(1e-12)
    aq = (a_blk / s_a_safe[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
    # weight: 128x128 blocks over [N, K] -> scale_b [KB, NB]
    NB = N // 128
    w_blk = wgt.float().reshape(NB, 128, KB, 128)                   # [NB, n_in_blk, KB, k_in_blk]
    s_w = w_blk.abs().amax(dim=(1, 3), keepdim=False) / 448.0       # [NB, KB]
    s_w_safe = s_w.clamp_min(1e-12)
    wq = (w_blk / s_w_safe[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)
    # required: scale_a (M, KB) contiguous, scale_b (KB, NB) outer-dim-major
    scale_a = s_a_safe.t().contiguous().t()  # [M, KB] outer-dim-major = stride (1, M)
    scale_b = s_w_safe.t().contiguous()      # [KB, NB] contiguous
    out = torch._scaled_mm(aq, wq.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    print(f"blockwise 1x128   : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print(f"blockwise 1x128   : FAIL -> {str(e)[:300]}")

# 3) MXFP8 (1x32 blocks, e8m0 scales)
try:
    from torchao.float8.inference import to_mxfp8  # may not exist in this torch
    print("torchao mxfp8 available")
except Exception:
    pass
# native torch mxfp8 path
try:
    from torch.nn.functional import scaled_mm  # torch 2.9+? check
except Exception:
    pass
try:
    import torch._C
    # e8m0 scales: use torch.float8_e8m0fnu
    KB32 = K // 32
    a_blk = g.float().reshape(M, KB32, 32)
    amax = a_blk.abs().amax(dim=-1)                                  # [M, KB32]
    e8m0_max = 2.0 ** 127
    s_a = (amax / 448.0)
    # e8m0: round scale UP to next power of two, express as exponent
    exp_a = torch.ceil(torch.log2(s_a.clamp_min(2 ** -127)))
    s_a_e8m0 = torch.exp2(exp_a).to(torch.float8_e8m0fnu)
    aq = (a_blk / torch.exp2(exp_a)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
    print("mxfp8 manual quant built; attempting scaled_mm...")
    NB32 = N // 32
    w_blk = wgt.float().reshape(NB32, 32)
    s_w = w_blk.abs().amax(dim=-1) / 448.0
    exp_w = torch.ceil(torch.log2(s_w.clamp_min(2 ** -127)))
    s_w_e8m0 = torch.exp2(exp_w).to(torch.float8_e8m0fnu)
    wq = (w_blk / torch.exp2(exp_w)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)
    out = torch._scaled_mm(aq, wq.t(), scale_a=s_a_e8m0, scale_b=s_w_e8m0, out_dtype=torch.bfloat16)
    print(f"mxfp8 1x32        : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print(f"mxfp8 1x32        : FAIL -> {str(e)[:200]}")