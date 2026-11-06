#!/bin/bash
# t3_support_sweep.sh — step 4 idea #2: sweep the naming support gate.
#
# Same checkpoints, same position-aware ranking, only min_support changes:
#   0.01 (already done as the "pa" leg), 0.05, 0.10
# Chosen on VALIDATION only. Nothing about the model changes.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_NAMER=pa
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
export T3A_K=32
export T3A_LR=1.0
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/support_sweep.log

echo "=== step 4 idea #2: support-gate sweep (K=32)  $(date '+%F %T') ===" >> "$LOG"
for MS in 0.05 0.10 0.20; do
  echo "--- min_support=$MS  $(date '+%T')" >> "$LOG"
  T3A_MIN_SUPPORT=$MS nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
  echo "    min_support=$MS rc=$?" >> "$LOG"
done
echo "=== support sweep done  $(date '+%F %T') ===" >> "$LOG"