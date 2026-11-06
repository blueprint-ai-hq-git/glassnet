#!/bin/bash
# Build a CLEAN val split for prepared_mix (the corpus the current run trains on) and re-measure the finished
# model on it. 48,615 val windows in the current split appear verbatim in train, so every accuracy figure
# measured on that split is partly a memorisation score. The fixed-point repair removes them; the model does
# not need retraining, because scoring a fixed model on a val that is disjoint from its train set is honest.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/CLEAN-VAL-MIX.txt
{
  echo "=== CLEAN VAL FOR prepared_mix $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_mix data/chat/prepared_mix_honest 32
  echo "repair exit=$?"
  echo "--- exact-byte confirmation on the repaired corpus"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/leak_exact_check.py \
    data/chat/prepared_mix_honest 32
  echo "check exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
