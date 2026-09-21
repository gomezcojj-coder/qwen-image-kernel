# Recover t2i results from the kernel benchmark log (UTF-16 PowerShell redirect).
import json
import re

txt = open("bench_out/big/kernel_log.txt", encoding="utf-16", errors="ignore").read()
pat = re.compile(r"A t2i (\d+)px p(\d): ([\d.]+)s \| ([\d.]+)s/step \| ([\d.]+) GiB")
rows = [{"cfg": "A", "res": int(m.group(1)), "prompt": int(m.group(2)),
         "total_s": float(m.group(3)), "mean_step_s": float(m.group(4)), "vram_gib": float(m.group(5))}
        for m in pat.finditer(txt)]
cold = re.search(r"cold start.*?: ([\d.]+)s", txt)
det = "determinism (same seed, bytes identical): True" in txt
out = {"t2i": rows, "edit": [], "meta": {"gpu": "NVIDIA GeForce RTX 5070", "torch": "2.14.0+cu130"},
       "cold_start_s": float(cold.group(1)) if cold else None, "determinism_bitexact": det}
json.dump(out, open("bench_out/kernel_bench.json", "w"), indent=2)
print("recovered", len(rows), "t2i rows; cold =", out["cold_start_s"], "s; deterministic =", det)