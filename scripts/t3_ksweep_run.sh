#!/bin/bash
# t3_ksweep_run.sh — step 2 phase 2: for every K, train BOTH models for 3 seeds
# at the lr the FROZEN JOINT SELECTION RULE chose, then re-audit the readable.
#
# Selections come from runs/t3/ksweep_selection.json (written by
# t3_ksweep_pick.py) — read at run time so they cannot drift from what was
# frozen. Checkpoints are saved by t3_models.py and LOADED by the audit, so the
# audit runs on the exact models whose numbers are reported.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/ksweep.log

echo "=== K sweep phase 2 (3 seeds at the frozen selection)  $(date '+%F %T') ===" >> "$LOG"

# --- training: both models, 3 seeds, at the selected lr ---
while read -r K MODEL LR; do
  echo "--- train K=$K $MODEL lr=$LR  $(date '+%T')" >> "$LOG"
  T3_K=$K T3_LR=$LR T3_MODELS=$MODEL T3_SEEDS=0,1,2 T3_TAG="K${K}_sel" \
    nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
  echo "    train K=$K $MODEL rc=$?" >> "$LOG"
done <<'EOF'
8 readable 1.0
8 normal 0.3
16 readable 1.0
16 normal 1.0
24 readable 0.5
24 normal 0.3
32 readable 1.0
32 normal 0.3
EOF

# --- audit the readable at every K, loading the saved checkpoints ---
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_BACKEND=cupy
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
while read -r K LR; do
  echo "--- audit K=$K lr=$LR  $(date '+%T')" >> "$LOG"
  T3A_K=$K T3A_LR=$LR \
    nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
  echo "    audit K=$K rc=$?" >> "$LOG"
done <<'EOF'
8 1.0
16 1.0
24 0.5
32 1.0
EOF

echo "=== phase 2 done  $(date '+%F %T') ===" >> "$LOG"