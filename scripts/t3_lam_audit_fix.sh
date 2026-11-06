#!/bin/bash
# t3_lam_audit_fix.sh — re-run the idea-#3 audits with lam_act now IN the output
# filename. The first attempt collided (four lam audits shared two filenames and
# also clashed with the earlier lam=0.2 file), so those JSONs are not trustworthy.
#
# Also re-runs the lam=0.2 K=32 lr=1.0 baseline so every leg has a clean,
# separately-named file. Checkpoints are loaded, so no model is retrained.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_NAMER=pa
export T3A_MIN_SUPPORT=0.01
export T3A_K=32
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/lam_audit_fix.log

echo "=== idea #3 audits, fixed filenames  $(date '+%F %T') ===" >> "$LOG"

# lam=0.2 baseline (its old JSON may have been clobbered) then the two new lams
for LAM in 0.2 0.5 1.0; do
  for LR in 0.5 1.0; do
    if [ "$LAM" = "0.2" ] && [ "$LR" = "0.5" ]; then
      continue      # lam=0.2 was only ever run at lr 1.0 at K=32
    fi
    echo "--- audit lam=$LAM lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_A=$LAM T3A_LR=$LR \
      nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
    echo "    audit lam=$LAM lr=$LR rc=$?" >> "$LOG"
  done
done
echo "=== fixed audits done  $(date '+%F %T') ===" >> "$LOG"