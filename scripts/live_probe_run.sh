#!/bin/bash
# live_probe_run.sh — run the audit with the METER-02 LIVE lie probe.
#
# Loads the already-trained readable checkpoints (T3A_LOAD_DIR) so nothing is
# retrained except the probe itself, then runs the live probe: the probe column
# trains NORMALLY until its val firing rate reaches 2%, and only THEN are its
# gradients frozen. Verdict per METER-RULE-02.md section 3.
#
# Runs it twice for comparison: under supervision (lc=1.0) and at baseline (lc=0),
# because the OLD probe was CAUGHT at baseline and MISSED under supervision.
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
# Meter-02: primary floor + the live-probe target firing rate
export T3A_FIRING_FLOOR=0.02
export T3A_LIVE_TARGET=0.02
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/live_probe.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY"

echo "=== METER-02 LIVE LIE PROBE  $(date '+%F %T') ===" >> "$LOG"

# ---- supervised (lam_concept=1.0, lr=0.5) ----
echo "--- supervised lc=1.0 lr=0.5" >> "$LOG"
T2_LAM_CONCEPT=1.0 T3A_LR=0.5 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py \
  >> "$LOG" 2>&1
RC=$?
echo "    lc=1.0 rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "live probe (supervised) failed rc=$RC"

# ---- baseline (lam_concept=0, lr=1.0) ----
echo "--- baseline lc=0 lr=1.0" >> "$LOG"
T2_LAM_CONCEPT=0 T3A_LR=1.0 T3A_N_SUP=0 nice -n 19 ionice -c3 "$PY" -u \
  scripts/t3_audit.py >> "$LOG" 2>&1
RC=$?
echo "    lc=0 rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "live probe (baseline) failed rc=$RC"

echo "=== LIVE PROBE DONE  $(date '+%F %T') ===" >> "$LOG"