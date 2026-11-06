#!/bin/bash
# run_target_i.sh — TARGET I: is the activation sparsity penalty why the two commonest concepts fail?
#
# One change from Target G2: lam_act 0.2 -> 0.0, at D=64 only (the reference rung), 3 seeds.
# Control arm: the same run at lam_act=0.2, which must reproduce G2's 7/9 with the same two failing.
#
# Order: audit control first (abort if it fails), then the two arms. Outputs are tagged so the
# arms cannot overwrite each other.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export T2_CONCEPT_SET=meaningful_ext
export T2_N_TAUGHT=9
export T2_LAM_CONCEPT=0.1
export G2_RUNGS=64
export G2_SEEDS=0,1,2
export G2_EPOCHS=60
PY=".venv-gpu/bin/python3"
LOG=runs/t3/target_i.log

echo "=== 0. AUDIT CONTROL (must reproduce Target E's 7/9) $(date '+%T') ===" | tee "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/g2_audit_verify.py 2>&1 \
  | sed -n '/CONTROL — does the G2 audit/,$p' | tee -a "$LOG" | tail -4
if ! grep -q "CONTROL: PASS" "$LOG"; then
  echo "CONTROL FAILED — refusing to run Target I." | tee -a "$LOG"; exit 1
fi

echo | tee -a "$LOG"
echo "=== ARM A: lam_act=0.0  (the one change) $(date '+%T') ===" | tee -a "$LOG"
G2_LAM_ACT=0.0 G2_OUT=target_i_lam0.json nice -n 19 ionice -c3 "$PY" -u scripts/g2_scaling.py 2>&1 \
  | grep -E "concept set gate|^  D=|SUMMARY|^    [0-9]+ |readable (PASS|FAIL)" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== ARM B (CONTROL): lam_act=0.2  (must reproduce G2's 7/9) $(date '+%T') ===" | tee -a "$LOG"
G2_LAM_ACT=0.2 G2_OUT=target_i_lam02.json nice -n 19 ionice -c3 "$PY" -u scripts/g2_scaling.py 2>&1 \
  | grep -E "concept set gate|^  D=|SUMMARY|^    [0-9]+ |readable (PASS|FAIL)" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== VERDICT $(date '+%T') ===" | tee -a "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/target_i_verdict.py 2>&1 | tee -a "$LOG"