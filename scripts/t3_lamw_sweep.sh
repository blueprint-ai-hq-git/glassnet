#!/bin/bash
# t3_lamw_sweep.sh — step 4 idea #4: raise lam_w (L1 on concept WEIGHTS).
#
# ONE CHANGE: lam_w 1.0 -> {2.0, 5.0}. lam_act stays 0.2. K=32.
# Rationale: weight sparsity concentrates each concept on fewer input dims, which
# should raise a word's contribution share WITHOUT reducing the fired-window count
# that the naming depends on (the failure mode of idea #3).
#
# Budget identical for both lam_w values: lr in {0.5,1.0} x 1 seed (frozen joint
# rule), then 3 seeds at the selected lr, then re-audit.
# T2_LAM_W is read at import by run_t2_iter003, so it must be exported before the
# interpreter starts.
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
LOG=runs/t3/lamw_sweep.log

echo "=== step 4 idea #4: lam_w sweep (K=32)  $(date '+%F %T') ===" >> "$LOG"

for LAMW in 2.0 5.0; do
  for LR in 0.5 1.0; do
    echo "--- tune lam_w=$LAMW lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_W=$LAMW T3_LR=$LR T3_SEEDS=0 T3_TAG="K32_lamw${LAMW}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    tune lam_w=$LAMW lr=$LR rc=$?" >> "$LOG"
  done
done

for LAMW in 2.0 5.0; do
  for LR in 0.5 1.0; do
    echo "--- confirm lam_w=$LAMW lr=$LR (3 seeds)  $(date '+%T')" >> "$LOG"
    T2_LAM_W=$LAMW T3_LR=$LR T3_SEEDS=0,1,2 T3_TAG="K32_lamw${LAMW}_lr${LR}_sel" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    confirm lam_w=$LAMW lr=$LR rc=$?" >> "$LOG"
  done
done

# --- audit: checkpoints are now lam_w-tagged, so load them (faster, and audits
#     the exact model whose accuracy/perplexity is reported) ---
export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_NAMER=pa
export T3A_MIN_SUPPORT=0.01
export T3A_K=32
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
for LAMW in 2.0 5.0; do
  for LR in 0.5 1.0; do
    echo "--- audit lam_w=$LAMW lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_W=$LAMW T3A_LR=$LR \
      nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
    echo "    audit lam_w=$LAMW lr=$LR rc=$?" >> "$LOG"
  done
done

echo "=== idea #4 done  $(date '+%F %T') ===" >> "$LOG"