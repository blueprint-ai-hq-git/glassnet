#!/bin/bash
# Re-run the repair with the fixed-point loop, so the val set reaches ZERO overlap with train.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== REPAIR FIXED POINT $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/REPAIR-CTRL.txt 2>&1
