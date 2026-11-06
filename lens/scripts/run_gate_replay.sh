#!/bin/bash
# run_gate_replay.sh — RUNNER for lens/scripts/gate_replay.py (predictions P109a-i).
# Predictions were written BEFORE the script existed: lens/runs/GATE-ARM-GRADE-PREDICTIONS-P109.txt.
#
# WHAT IT DOES: text-only offline replay of the shipped gap-gate rule on the LIVE gate arm's own per-step log
# (lens/runs/chat/gate_sgd04_s2_gates.jsonl). No model, no corpus, no GPU, no training, no engine edit.
# SAFE TO RUN BESIDE THE TRAINER: it opens one text file and prints.
#
# OUTPUT: appended (never overwritten) to lens/runs/GATE-REPLAY.txt
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUTF=lens/runs/GATE-REPLAY.txt
{
  echo "=============================================================================================="
  echo "GATE REPLAY RUNNER  $(date '+%F %H:%M:%S')"
  echo "=============================================================================================="
  nice -n 19 ionice -c3 python3 lens/scripts/gate_replay.py "$@"
  echo "REPLAY EXIT=$?"
} >> "$OUTF" 2>&1
