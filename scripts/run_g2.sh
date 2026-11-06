#!/bin/bash
# run_g2.sh — TARGET G2: readability across model sizes. Rungs differ only in width D.
#
# Order matters: the audit CONTROL runs FIRST. If the G2 audit cannot reproduce Target E's known
# 7/9 on E's checkpoint, the script aborts. The first version of this audit used the wrong
# contribution formula (|h*W| instead of the meter's logit-gain) and reported 0/9 at D=32, which
# was meaningless. Then a smoke run, so any remaining bug costs seconds instead of an hour.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
# THE CONCEPT SET MUST BE SET EXPLICITLY. The loader defaults to "words" (single-word targets),
# which is NOT what the audit compares against. The first G2 sweep omitted these two lines and
# produced a completely invalid result (every concept near chance).
export T2_CONCEPT_SET=meaningful_ext
export T2_N_TAUGHT=9
export T2_LAM_CONCEPT=0.1
mkdir -p runs/t3/ckpt
LOG=runs/t3/g2_scaling.log
PY=".venv-gpu/bin/python3"

echo "=== 0. AUDIT CONTROL (must reproduce Target E's 7/9) $(date '+%T') ===" | tee "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/g2_audit_verify.py 2>&1 \
  | sed -n '/CONTROL — does the G2 audit/,$p' | tee -a "$LOG" | tail -8
if ! grep -q "CONTROL: PASS" "$LOG"; then
  echo "CONTROL FAILED — refusing to run G2 with an unverified audit." | tee -a "$LOG"
  exit 1
fi

echo | tee -a "$LOG"
echo "=== 1. SMOKE (D=32, 1 seed, 3 epochs) $(date '+%T') ===" | tee -a "$LOG"
G2_RUNGS=32 G2_SEEDS=0 G2_EPOCHS=3 nice -n 19 ionice -c3 "$PY" -u \
  scripts/g2_scaling.py 2>&1 | tail -16 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2. G2 FULL (D=32,64,128; 3 seeds; 60 epochs) $(date '+%T') ===" | tee -a "$LOG"
G2_RUNGS=32,64,128 G2_SEEDS=0,1,2 G2_EPOCHS=60 nice -n 19 ionice -c3 "$PY" -u \
  scripts/g2_scaling.py 2>&1 | tee -a "$LOG" | tail -48