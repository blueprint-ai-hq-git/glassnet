#!/bin/bash
# SCENARIO BATTERY, corrected: three ships through the SAME runtime path, one after another.
# The previous runs all loaded t11 because the runtime hard-pinned CHAT_FINAL and ignored
# TALK_CKPT -- so the "bigger model word salad" was really the old model's output. Each ship is
# now loaded explicitly via TALK_SHIP (which variants.py honours).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
for ship in t11_talk41_ep192 t12_d512_full_ep192 t12_splitfix_d512_ep192; do
  echo "########## SHIP $ship"
  TALK_SHIP="$PWD/lens/runs/chat/${ship}.npz" bash lens/scripts/run_talk_scenarios.sh "$ship" \
    | grep -E "ship:|PROMPT|REPLY|STATS"
done