#!/bin/bash
# t3_lam_sweep.sh — step 4 idea #3: raise lam_act (concept-activation sparsity).
#
# ONE CHANGE: lam_act 0.2 -> {0.5, 1.0}. Everything else identical.
# Budget, identical for both lam values: lr in {0.5, 1.0} x 1 seed (chosen on
# validation by the frozen joint rule), then 3 seeds at the selected lr, then
# re-audit with the position-aware namer at support 1%.
#
# Checkpoints are lam-tagged by t3_models.py, so nothing existing is overwritten.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=32
export T3_MODELS=readable,normal
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/lam_sweep.log

echo "=== step 4 idea #3: lam_act sweep (K=32)  $(date '+%F %T') ===" >> "$LOG"

# --- phase 1: tune lr for each new lam_act, 1 seed, both models ---
for LAM in 0.5 1.0; do
  for LR in 0.5 1.0; do
    echo "--- tune lam=$LAM lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_A=$LAM T3_LR=$LR T3_SEEDS=0 T3_TAG="K32_lam${LAM}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    tune lam=$LAM lr=$LR rc=$?" >> "$LOG"
  done
done

# --- phase 2: 3 seeds at each new lam_act's selected lr (lr 1.0 was best at
#     K=32 for lam=0.2; the tuning above picks per-lam) ---
for LAM in 0.5 1.0; do
  for LR in 0.5 1.0; do
    echo "--- confirm lam=$LAM lr=$LR (3 seeds)  $(date '+%T')" >> "$LOG"
    T2_LAM_A=$LAM T3_LR=$LR T3_SEEDS=0,1,2 T3_TAG="K32_lam${LAM}_lr${LR}_sel" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    confirm lam=$LAM lr=$LR rc=$?" >> "$LOG"
  done
done

# --- phase 3: audit the readable at each new lam_act ---
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_BACKEND=cupy
export T3A_NAMER=pa
export T3A_MIN_SUPPORT=0.01
export T3A_K=32
export T3A_SEEDS=0,1,2
for LAM in 0.5 1.0; do
  for LR in 0.5 1.0; do
    echo "--- audit lam=$LAM lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_A=$LAM T3A_LR=$LR \
      nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
    echo "    audit lam=$LAM lr=$LR rc=$?" >> "$LOG"
  done
done

echo "=== idea #3 done  $(date '+%F %T') ===" >> "$LOG"