#!/bin/bash
# RUNNER for the gate-cost grader (text only: no checkpoint, no model, no GPU, no training).
# Predictions: lens/runs/GATE-COST-GRADE-PREDICTIONS-P97.txt (P97a/b/c/g), written before the grader existed.
# Graded run's own predictions: lens/runs/GATE-COST-PREDICTIONS-P87.txt (P87a-h).
# The grader writes lens/runs/GRADE-P87.txt itself; this runner writes lens/runs/GRADE-P87-RUN.txt.
# It is run SECOND, after lens/scripts/run_padfree_gatecost.sh, so that P87e/P87g are graded in the same pass.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "=== GATE-COST GRADER $(date '+%F %H:%M:%S') ==="
  echo "inputs (all read-only):"
  stat -c '  %y  %s bytes  %n' lens/runs/GATE-COST.txt \
    lens/runs/chat/ours3M_ctrl_std_history.jsonl \
    lens/runs/chat/ours3M_relgrad_nogates_history.jsonl \
    lens/runs/chat/ours_d256_3M_2ep_history.jsonl \
    lens/runs/PADFREE-TALK-GATECOST.txt
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/gate_cost_numbers.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/GRADE-P87-RUN.txt 2>&1
