#!/bin/bash
# a2_probe.sh — A2-2: the lie probe under A2 settings, caught while live.
# Trains the probe (freeze column 2 -> it is never supervised on its target) and saves
# its checkpoint, then audits it with Meter-03 against the target it was never trained on.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_CONCEPT_SET=clusters32
export T2_N_TAUGHT=24
export T2_LAM_CONCEPT=0.5
export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_NAMER=pa
export T3A_MIN_SUPPORT=0.01
export T3A_K=32
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
export T3A_N_SUP=24
export T3A_LIE_COL=2
export T3A_FIRING_FLOOR=0.02
export T3A_LIVE_TARGET=0.02
export T3A_FORCE_PROBE=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a2_run.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
echo "--- A2 lie probe (train + save checkpoint)  $(date '+%T')" >> "$LOG"
T3A_LR=0.5 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py >> "$LOG" 2>&1
RC=$?
echo "    a2 probe rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "A2 probe run failed rc=$RC"

echo "--- A2 probe: Meter-03 verdict vs the target it was never trained on" >> "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_m3_probe_a2.py 2>&1 | tee -a "$LOG"
RC=$?
echo "    a2 probe audit rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "A2 probe audit failed rc=$RC"
echo "=== A2 PROBE DONE  $(date '+%F %T') ===" >> "$LOG"