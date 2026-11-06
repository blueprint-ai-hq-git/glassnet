#!/bin/bash
# t3_namer_ab.sh — step 4 idea #1 A/B: audit the SAME saved checkpoints with the
# OLD namer and the NEW position-aware namer.
#
# The old-namer leg must reproduce the K-sweep numbers exactly (26.4% at K=24,
# 22.9% at K=32); if it does not, the comparison setup is invalid and the new
# leg means nothing. Nothing about the model changes — same checkpoints loaded.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/namer_ab.log

echo "=== step 4 idea #1: namer A/B  $(date '+%F %T') ===" >> "$LOG"

# K=24 (readable lr 0.5) and K=32 (readable lr 1.0)
for LEG in old pa; do
  for KLR in "24 0.5" "32 1.0"; do
    set -- $KLR
    K=$1; LR=$2
    echo "--- namer=$LEG K=$K lr=$LR  $(date '+%T')" >> "$LOG"
    T3A_NAMER=$LEG T3A_K=$K T3A_LR=$LR \
      nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
    echo "    namer=$LEG K=$K rc=$?" >> "$LOG"
  done
done
echo "=== A/B done  $(date '+%F %T') ===" >> "$LOG"