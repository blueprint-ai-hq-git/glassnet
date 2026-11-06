#!/bin/bash
# sweep_lc.sh — find the cheapest lam_concept that still gets 9/9 concepts.
#
# FIX-01 established the two ends of the trade-off:
#   lam_concept=0.1 -> 7/9 concepts pass, gap +12.281
#   lam_concept=1.0 -> 9/9 concepts pass, gap +38.025
# This sweeps the middle. Each arm writes to its own directory; nothing is overwritten.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_ONLY=readable
export BLOCKER_EPOCHS="${SWEEP_EPOCHS:-60}"

for lc in 0.3 0.5; do
  echo "=============================================================="
  echo "SWEEP lam_concept=$lc"
  echo "=============================================================="
  BLOCKER_LC=$lc BLOCKER_OUT="fix01_lc${lc}" bash lens/scripts/run_blocker.sh 2>&1 \
    | grep -E "DONE|concepts |GAP|readable |normal |^    [a-z_]+ +AUROC" | tail -16
done

echo
echo "=============================================================="
echo "SWEEP SUMMARY"
echo "=============================================================="
.venv-gpu/bin/python3 - <<'PY'
import json, os
PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
rows = []
for tag, p in [("0.1", "blocker"), ("0.3", "fix01_lc0.3"), ("0.5", "fix01_lc0.5"),
               ("1.0", "fix01_lc1")]:
    f = os.path.join(PROJ, "lens", "runs", p, "readable_summary.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    rows.append((tag, d["n_pass"], d["n_concepts"], d["val_ppl"], d["val_acc"]))
norm = json.load(open(os.path.join(PROJ, "lens", "runs", "blocker", "normal_summary.json")))
print(f"  matched normal ppl {norm['val_ppl']:.3f}")
print(f"  {'lam_concept':<13}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
for tag, np_, nc, ppl, acc in rows:
    print(f"  {tag:<13}{f'{np_}/{nc}':>10}{ppl:>10.3f}{ppl-norm['val_ppl']:>+10.3f}{acc*100:>8.2f}%")
PY