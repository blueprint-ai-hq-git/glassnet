#!/bin/bash
# run_orderblind.sh -- THE ORDER-BLIND TEST (predictions P117a-g)
# Predictions were written BEFORE lens/scripts/orderblind_test.py existed:
#   lens/runs/ORDER-BLIND-PREDICTIONS-P117.txt
# CPU ONLY: no training, no generation, no GPU, no checkpoint written; every input is read-only.
# It runs on cores 2,3 at nice 19 because a GPU training arm owns cores 0,1 in this project.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export EVAL_DEVICE=cpu
export PYTHONUNBUFFERED=1
OUT=lens/runs/ORDER-BLIND.txt
{
  echo "####################################################################################################"
  echo "# ORDER-BLIND TEST run $(date '+%F %H:%M:%S')  scope=${OB_SCOPE:-manual}"
  echo "# predictions: lens/runs/ORDER-BLIND-PREDICTIONS-P117.txt (written before the script existed)"
  echo "# APPEND-ONLY: an earlier block in this file stays on the record."
  echo "####################################################################################################"
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/orderblind_test.py
  echo "# exit=$?"
} >> "$OUT" 2>&1
tail -40 "$OUT"
