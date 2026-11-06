#!/bin/bash
# a2_margin.sh — A2-4: readable vs matched normal on accuracy AND perplexity (3 seeds).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
"$PY" - <<'EOF'
import json, math
import numpy as np
p = "runs/t3/models_K32_a2.json"
rows = json.load(open(p))
r = [x for x in rows if x["model"] == "readable"]
n = [x for x in rows if x["model"] == "normal"]
def st(rs, k):
    v = [x[k] for x in rs]
    return float(np.mean(v)), float(np.std(v, ddof=1))
ra, rs_ = st(r, "val"); rp, rps = st(r, "val_ppl")
na, ns = st(n, "val"); np_, nps = st(n, "val_ppl")
dacc, dppl = ra - na, np_ - rp
pool_a = math.sqrt(rs_**2 + ns**2); pool_p = math.sqrt(rps**2 + nps**2)
print("=== A2-4: readable vs matched normal (3 seeds, validation) ===")
print(f"  readable : acc {ra:.2f}±{rs_:.2f}   ppl {rp:.2f}±{rps:.2f}")
print(f"  normal   : acc {na:.2f}±{ns:.2f}   ppl {np_:.2f}±{nps:.2f}")
print(f"  dacc {dacc:+.2f} vs 2*pooled {2*pool_a:.3f}  -> "
      f"{'OK' if dacc >= -2*pool_a else 'FAIL'}")
print(f"  dppl {dppl:+.2f} vs 2*pooled {2*pool_p:.3f}  -> "
      f"{'OK' if dppl >= -2*pool_p else 'FAIL'}")
ok = (dacc >= -2*pool_a) and (dppl >= -2*pool_p)
print(f"  A2-4 (readable no worse on BOTH): {'PASS' if ok else 'FAIL'}")
print(f"  seeds: readable {[round(x['val'],2) for x in r]}")
print(f"         normal   {[round(x['val'],2) for x in n]}")
json.dump({"dacc": dacc, "dppl": dppl, "pool_a": pool_a, "pool_p": pool_p,
           "A2_4_pass": bool(ok)}, open("runs/t3/a2_margin.json", "w"), indent=1)
EOF