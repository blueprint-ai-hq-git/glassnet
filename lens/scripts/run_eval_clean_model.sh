#!/bin/bash
# Honest accuracy for the finished clean-corpus model, measured on the DISJOINT val.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
SHIP=$(ls -t lens/runs/chat/cl_d256_clean_24_ep24.npz 2>/dev/null | head -1)
{
  echo "=== EVAL CLEAN MODEL $(date '+%F %H:%M')  ship $(basename "$SHIP")"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" bash lens/scripts/run_eval_clean.sh "$SHIP"
  echo "exit=$?"
} >> lens/runs/EVAL-CLEAN.txt 2>&1
