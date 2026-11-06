#!/bin/bash
# PAD-FREE TALK TEST v2 (predictions P71a-f written BEFORE: lens/runs/PADFREE-TALK-PREDICTIONS-P71.txt).
# Reads the stored talk logs, writes ONLY lens/runs/PADFREE-TALK.txt. CPU only, nice 19.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK.txt"
{
  echo "=== PAD-FREE TALK v2 runner $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v2_padfree.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/PADFREE-TALK-RUN.txt 2>&1
