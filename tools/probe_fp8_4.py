# Probe 4: correct scale pairing + find the blockwise stride convention.
import itertools
import torch

print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.manual_seed(0)

M, K, N = 1024, 4096, 4096
g = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
wgt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
ref = g.float() @ wgt.float().t()

def relerr(out):
    return ((out.float() - ref).norm() / ref.norm()).item()

# 1) tensorwise, correctly paired
try:
    s_a = (g.abs().amax() / 448.0).float().reshape(1)
    s_b = (wgt.abs().amax() / 448.0).float().reshape(1)
    aq = (g / s_a).clamp(-448, 448).to(torch.float8_e4m3fn)
    wq = (wgt / s_b).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(aq, wq.t(), scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    print(f"tensorwise      : OK  rel-err {relerr(out):.5f}")
except Exception as e:
    print("tensorwise      : FAIL ->", str(e)[:150])

# 2) blockwise 1x128 x 128x128, sweep stride conventions
KB, NB = K // 128, N // 128
a_blk = g.float().reshape(M, KB, 128)
s_a = (a_blk.abs().amax(dim=-1) / 448.0).float().clamp_min(1e-12)      # [M, KB]
aq = (a_blk / s_a[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
w_blk = wgt.float().reshape(NB, 128, KB, 128)
s_w = (w_blk.abs().amax(dim=(1, 3)) / 448.0).float().clamp_min(1e-12)  # [NB, KB]
wq = (w_blk / s_w[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)

# scale_a candidates: [M, KB] with row-major or col-major layout
cands_a = {
    "a row-major (KB,1)": s_a.contiguous(),
    "a col-major (1,M) ": s_a.t().contiguous().t(),
}
# scale_b candidates: [KB, NB] built from s_w [NB, KB]
cands_b = {
    "b row-major (NB,1)": s_w.t().contiguous(),
    "b col-major (1,NB)": s_w.t().contiguous().t(),
    "b as [NB,KB] rm    ": s_w.contiguous(),
}
for (na, ca), (nb, cb) in itertools.product(cands_a.items(), cands_b.items()):
    try:
        out = torch._scaled_mm(aq, wq.t(), scale_a=cands_a[na], scale_b=cands_b[nb], out_dtype=torch.bfloat16)
        print(f"blockwise {na} + {nb}: OK  rel-err {relerr(out):.5f}")
        break
    except Exception as e:
        msg = str(e).split("\n")[0][:110]
        print(f"blockwise {na} + {nb}: FAIL -> {msg}")

# 3) MXFP8 1x32 e8m0 (fixed)
try:
    KB32, NB32 = K // 32, N // 32
    a_blk = g.float().reshape(M, KB32, 32)
    exp_a = torch.ceil(torch.log2((a_blk.abs().amax(dim=-1) / 448.0).clamp_min(2.0 ** -127)))
    s_a8 = torch.exp2(exp_a).to(torch.float8_e8m0fnu)                     # [M, KB32]
    aq2 = (a_blk / torch.exp2(exp_a)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
    w_blk = wgt.float().reshape(N, KB32, 32)
    exp_w = torch.ceil(torch.log2((w_blk.abs().amax(dim=-1) / 448.0).clamp_min(2.0 ** -127)))
    s_w8 = torch.exp2(exp_w).to(torch.float8_e8m0fnu)                     # [N, KB32]
    wq2 = (w_blk / torch.exp2(exp_w)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(N, K)
    ok = False
    for name, (sca, scb) in {
        "[M,KB32]rm + [N,KB32]rm.t()": (s_a8, s_w8.reshape(NB32, 32).t().contiguous()),
        "a.t().c().t() + b.t().c()  ": (s_a8.t().contiguous().t(), s_w8.reshape(NB32, 32).t().contiguous()),
    }.items():
        try:
            out = torch._scaled_mm(aq2, wq2.t(), scale_a=sca, scale_b=scb, out_dtype=torch.bfloat16)
            print(f"mxfp8 {name}: OK  rel-err {relerr(out):.5f}")
            break
        except Exception as e:
            print(f"mxfp8 {name}: FAIL -> {str(e)[:130]}")
except Exception as e:
    print(f"mxfp8           : FAIL -> {str(e)[:130]}")