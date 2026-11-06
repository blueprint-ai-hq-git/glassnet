#!/bin/bash
# a3_stage2.sh — TARGET A3 after tuning: matched normal, Meter-03 audit, lie probe, margin.
# Reads the winner from a3_selection.json so nothing can be chosen by hand after results.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=16
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T2_CONCEPT_SET=clusters16
export T2_N_TAUGHT=10
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a3_run.log
fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }

[ -f runs/t3/a3_selection.json ] || fail "no a3_selection.json"
LR=$("$PY" -c "import json;print(json.load(open('runs/t3/a3_selection.json'))['lr'])")
LC=$("$PY" -c "import json;print(json.load(open('runs/t3/a3_selection.json'))['lc'])")
LCS=$(echo "$LC" | tr -d '.')
echo "=== A3 STAGE 2 — selected lr=$LR lc=$LC  $(date '+%F %T') ===" | tee -a "$LOG"

# the readable 3 seeds already exist from the screen (seed 0) + confirm (seeds 1,2)
for s in 0 1 2; do
  [ -f "runs/t3/ckpt/readable_K16_lr${LR}_lc${LCS}_clusters16_seed${s}.npz" ] \
    || fail "missing readable checkpoint for seed $s"
done

echo "--- matched normal, same lr, same budget, 3 seeds" | tee -a "$LOG"
T2_LAM_CONCEPT="$LC" T3_LR="$LR" T3_SEEDS=0,1,2 T3_MODELS=normal T3_TAG=a3_normal \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "normal training failed"

echo "--- Meter-03 audit: 10 taught vs their targets, 6 untaught vs THEIR targets" \
  | tee -a "$LOG"
T3M_A3=1 T3M_K=16 T3M_N_SUP=10 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py \
  >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "Meter-03 audit failed"

echo "--- A3 lie probe (train + audit)" | tee -a "$LOG"
T3A_BACKEND=cupy T3A_LOAD_DIR="runs/t3/ckpt" T3A_NAMER=pa T3A_MIN_SUPPORT=0.01 \
T3A_K=16 T3A_SEEDS=0,1,2 T3A_EPOCHS=96 T3A_MIN_EPOCH=30 T3A_PATIENCE=15 \
T3A_N_SUP=10 T3A_LIE_COL=2 T3A_FIRING_FLOOR=0.02 T3A_LIVE_TARGET=0.02 \
T3A_FORCE_PROBE=1 T3A_LR="$LR" T2_LAM_CONCEPT="$LC" \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "probe training failed"

PROBE_CK="lieprobe_K16_lr${LR}_lc${LCS}_clusters16_seed1.npz"
T3P_KIND=clusters16 T3P_K=16 T3P_NSUP=10 T3P_CK="$PROBE_CK" T3P_COL=2 T3P_SEED=1 \
T3P_OUT=a3_lieprobe.json \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_m3_probe_x.py 2>&1 | tee -a "$LOG"
[ $? -ne 0 ] && fail "probe audit failed"

echo "=== A3 STAGE 2 DONE  $(date '+%F %T') ===" | tee -a "$LOG"