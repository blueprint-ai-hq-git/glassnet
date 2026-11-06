#!/bin/bash
# Verify the RUNNING system can actually serve the new 19/19 image ship: load it, answer one
# real image with see(), and answer one caption with the see+talk path -- the same calls the
# one-chat and combo matrix make. A ship that audits beautifully but crashes in the runtime is
# not a ship (the lesson of BUG-MED-1).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/one_chat.py \
  --ship seetalk_k21o_mirror.npz lens/runs/combos/one_chat_smoke_in.txt