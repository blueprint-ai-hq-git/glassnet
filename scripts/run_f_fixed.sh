#!/bin/bash
# run_f_fixed.sh — THE CORRECTED TARGET F CONCEPT EXPERIMENT.
#
# The first F run kept the snapshot at best LM perplexity, which on generated data is epoch 1,
# so its concepts were audited before they trained. This re-runs the readable student on the
# SAME generated corpus with F_CKPT_RULE=last (no early stop, all 40 epochs), then audits it
# with the corrected loader. This is the first honest answer to "do the concepts survive
# self-training?".
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T3M_IMPORT_ONLY=1
export T3M_K=9
PY=".venv-gpu/bin/python3"
LOG=runs/t3/f_fixed.log

echo "=== CORRECTED F: readable student, rule=all, ckpt=last, 40 epochs  $(date '+%T') ===" | tee -a "$LOG"
F_KIND=readable F_RULE=all F_SEED=0 F_LR=1.0 F_LC=0.1 F_EPOCHS=40 F_PATIENCE=99 \
  F_CKPT_RULE=last F_TAG=f2 \
  nice -n 19 ionice -c3 "$PY" -u scripts/f_train.py >> "$LOG" 2>&1
echo "training rc=$?" | tee -a "$LOG"
grep -E "^  ep|STUDENT" "$LOG" | tail -4

echo
echo "=== auditing the CORRECTLY saved student (f2) ===" | tee -a "$LOG"
sed 's/{rule}_f_seed/{rule}_f2_seed/' scripts/f_audit.py > scripts/f_audit_f2.py
grep -n "student_readable_{rule}" scripts/f_audit_f2.py
nice -n 19 ionice -c3 "$PY" -u scripts/f_audit_f2.py 2>&1 \
  | sed -n '/TARGET F — DO THE CONCEPTS/,$p' | tail -30