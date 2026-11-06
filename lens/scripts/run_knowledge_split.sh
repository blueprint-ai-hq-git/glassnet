#!/bin/bash
# Finish the knowledge corpus: create the base split, then make val disjoint from train at the training
# window (32 tokens), exactly as the other honest corpora were made. CPU only, nice 19.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=/tmp/knowledge_split.log
{
  echo "=== KNOWLEDGE SPLIT $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/make_split.py data/chat/prepared_kn 0.02 0
  echo "make_split exit=$?"
  echo
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_kn data/chat/prepared_kn_honest 32
  echo "repair exit=$?"
  echo
  echo "--- corpus sizes:"
  du -sh data/chat/prepared_kn data/chat/prepared_kn_honest 2>/dev/null
  echo "=== DONE $(date '+%F %H:%M')"
} > "$LOG" 2>&1
tail -20 "$LOG"