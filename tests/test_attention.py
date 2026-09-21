# Unit tests: Triton block-causal attention vs (a) a pure-torch reference and
# (b) the stock multi-pass SDPA decomposition used by QwenImage21AttnProcessor.
import sys

import torch

from qwen_image_kernel.triton_attention import block_causal_flash_attention, reference_block_causal_attention

torch.manual_seed(0)
DEV = "cuda"


def stock_reference(query, key, value, segments, prefix_len, key_valid=None):
    """The stock QwenImage21AttnProcessor prefill: per-segment SDPA calls.
    query [B, S, H, D]; segments [(s, e, is_text)] over the prefix; the target
    attends to everything."""
    B, S, H, D = query.shape
    outputs = []
    for start, end, is_text in segments:
        seg_mask = None
        if is_text:
            seg_len = end - start
            seg_mask = torch.cat(
                [
                    torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                    torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                ],
                dim=1,
            )[None, None]
        if key_valid is not None:
            kv = key_valid[:, None, None, :end]
            seg_mask = kv if seg_mask is None else (seg_mask & kv)
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end].transpose(1, 2),
                key[:, :end].transpose(1, 2),
                value[:, :end].transpose(1, 2),
                attn_mask=seg_mask,
            ).transpose(1, 2)
        )
    outputs.append(
        torch.nn.functional.scaled_dot_product_attention(
            query[:, prefix_len:].transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=key_valid[:, None, None, :] if key_valid is not None else None,
        ).transpose(1, 2)
    )
    return torch.cat(outputs, dim=1)


def make_segments(seq_layout):
    """seq_layout: list of ('text', n) / ('img', n) runs -> segments."""
    segs, pos = [], 0
    for kind, n in seq_layout:
        segs.append((pos, pos + n, kind == "text"))
        pos += n
    return segs, pos


def run_case(name, seq_layout, batch=1, heads=4, dim=128, use_padding=False, seq_scale=1.0):
    segs, S = make_segments(seq_layout)
    q = torch.randn(batch, S, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    k = torch.randn(batch, S, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    v = torch.randn(batch, S, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3

    key_valid = None
    if use_padding:
        # pad the first text run's tail (simulates right-padded prompt)
        first_text_end = next(e for s, e, t in segs if t)
        key_valid = torch.ones(batch, S, dtype=torch.bool, device=DEV)
        key_valid[:, first_text_end - 5 : first_text_end] = False

    L = torch.arange(S, dtype=torch.int32, device=DEV)
    for s, e, is_text in segs:
        if not is_text:
            L[s:e] = e - 1
    L[segs[-1][1]:] = S - 1  # target rows full attention

    mine = block_causal_flash_attention(q, k, v, L, key_valid)
    ref = reference_block_causal_attention(q, k, v, L, key_valid)
    stock = stock_reference(q, k, v, segs, segs[-1][1], key_valid)

    d_ref = (mine.float() - ref.float()).abs().max().item()
    d_stock = (mine.float() - stock.float()).abs().max().item()
    ok = d_ref < 2e-2 and d_stock < 2e-2
    print(f"{name:38s} max|diff| vs torch-ref {d_ref:.5f} | vs stock-SDPA {d_stock:.5f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_decode_case(name, prefix_len, target_len, batch=1, heads=4, dim=128):
    q = torch.randn(batch, target_len, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    k_new = torch.randn(batch, target_len, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    v_new = torch.randn(batch, target_len, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    ck = torch.randn(batch, prefix_len, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3
    cv = torch.randn(batch, prefix_len, heads, dim, device=DEV, dtype=torch.bfloat16) * 0.3

    KV = prefix_len + target_len
    L = torch.full((target_len,), KV - 1, dtype=torch.int32, device=DEV)
    mine = block_causal_flash_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
    ref = reference_block_causal_attention(q, k_new, v_new, L, None, cache_k=ck, cache_v=cv)
    d = (mine.float() - ref.float()).abs().max().item()
    print(f"{name:38s} max|diff| vs torch-ref {d:.5f} -> {'PASS' if d < 2e-2 else 'FAIL'}")
    return d < 2e-2


ok = True
ok &= run_case("t2i prefill (text+target)", [("text", 256), ("img", 1024)])
ok &= run_case("edit prefill (text+img+text+img)", [("text", 64), ("img", 4096), ("text", 32), ("img", 256)])
ok &= run_case("t2i with padded prompt", [("text", 128), ("img", 512)], use_padding=True)
ok &= run_case("batch=2", [("text", 96), ("img", 200)], batch=2)
ok &= run_decode_case("decode (prefix 256 + target 512)", 256, 512)
ok &= run_decode_case("decode (prefix 1024 + target 1024)", 1024, 1024)

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)