#!/bin/bash
# verify_b_machinery.sh — check the Target B evaluation machinery WITHOUT touching the grid.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY=".venv-gpu/bin/python3"

echo "=== 1. syntax check every touched script ==="
for f in scripts/t3_b_report.py scripts/t3_audit_m3.py scripts/t3_audit.py \
         scripts/t3_m3_probe_x.py scripts/build_handoff.py; do
  "$PY" -m py_compile "$f" && echo "  OK   py $f" || echo "  FAIL py $f"
done
for f in scripts/b_grid.sh scripts/b_audit.sh scripts/run_build_handoff.sh; do
  bash -n "$f" && echo "  OK   sh $f" || echo "  FAIL sh $f"
done

echo
echo "=== 2. does the report refuse to score a partial grid? (must exit 2) ==="
"$PY" scripts/t3_b_report.py 2>&1 | tail -4
echo "  exit code: ${PIPESTATUS[0]}  (2 = correctly refused)"

echo
echo "=== 3. is the checkpoint tag still separating A3 from B? ==="
ls -1 runs/t3/ckpt/readable_K16_lr1.0_lc01_clusters16_seed0.npz 2>&1 | sed 's/^/  A3 (untagged): /'
ls -1 runs/t3/ckpt/readable_K16_lr*_b_seed*.npz 2>/dev/null | head -3 | sed 's/^/  B (tagged):    /'
echo "  A3 probe still present: $(ls runs/t3/ckpt/lieprobe_K16_lr1.0_lc01_clusters16_seed1.npz 2>/dev/null | wc -l)"

echo
echo "=== 4. grid still running untouched? ==="
ps -eo pid,etime,cmd | grep "[t]3_models" | head -2
date '+  now %F %T'