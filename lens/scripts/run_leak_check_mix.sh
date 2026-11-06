#!/bin/bash
# Is the val set of the corpus being trained on actually clean? The run at epoch 13 measures masked accuracy
# on data/chat/prepared_mix's val, and that split never went through the fixed-point repair. Until this
# check says 0, the 33.40% figure must be treated as partly a memorisation score.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/LEAK-CHECK-MIX.txt
{
  echo "=== LEAK CHECK prepared_mix $(date '+%F %H:%M')"
  echo "exact bytes, not fingerprints: this is the authority on whether val text appears in train"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/leak_exact_check.py \
    data/chat/prepared_mix 32
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
