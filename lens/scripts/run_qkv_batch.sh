#!/bin/bash
# PLAN item (f3): does the batched [Wq|Wk|Wv] projection move a TOKEN? Identity, control, loop cost.
# PREDICTIONS, written before the run and before the code change: lens/runs/QKV-BATCH-PREDICTIONS.txt
# Output: lens/runs/QKV-BATCH.txt (appended, never overwritten) + lens/runs/qkv_batch_t{1,3}.json
# Short (a few minutes), so it is run in the foreground, not queued: the machine has a training arm
# running, and the ABSOLUTE times are therefore inflated - only the ratios inside one process are read.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "=== BATCHED QKV PROJECTION (f3) $(date '+%F %H:%M') ==="
  for T in 1 3; do
    QKV_BATCH_THREADS=$T nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/qkv_batch_test.py
    echo "exit=$? (threads=$T)"
  done
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/QKV-BATCH.txt 2>&1
