#!/bin/bash
# probe_ckpt_run.sh — re-run ONLY the lie probe so its checkpoint is saved, for the
# Meter-03 re-audit (which needs the probe's ACTIVATIONS, not just its Meter-02
# numbers). The per-seed audits are already recorded, so they are skipped.
# Probe settings are UNCHANGED from the Meter-02 run (same seeds, same freeze rule).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
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
export T3A_N_SUP=16
export T3A_LIE_COL=2
export T3A_FIRING_FLOOR=0.02
export T3A_LIVE_TARGET=0.02
export T3A_FORCE_PROBE=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/live_probe.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY"

echo "=== PROBE CHECKPOINT RUN (for Meter-03)  $(date '+%F %T') ===" >> "$LOG"

echo "--- supervised lc=1.0 lr=0.5" >> "$LOG"
T2_LAM_CONCEPT=1.0 T3A_LR=0.5 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py \
  >> "$LOG" 2>&1
RC=$?; echo "    lc=1.0 rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "probe (supervised) rc=$RC"

echo "--- baseline lc=0 lr=1.0" >> "$LOG"
T2_LAM_CONCEPT=0 T3A_LR=1.0 T3A_N_SUP=0 nice -n 19 ionice -c3 "$PY" -u \
  scripts/t3_audit.py >> "$LOG" 2>&1
RC=$?; echo "    lc=0 rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "probe (baseline) rc=$RC"

echo "--- probe checkpoints on disk:" >> "$LOG"
ls -1 runs/t3/ckpt/lieprobe_*.npz >> "$LOG" 2>&1
echo "=== PROBE CHECKPOINT RUN DONE  $(date '+%F %T') ===" >> "$LOG"