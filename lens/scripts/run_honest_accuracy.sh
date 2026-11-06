#!/bin/bash
# Honest accuracy for the finished full-corpus model: score gn_d256_full_24_ep24 on the val of
# prepared_mix_honest, which is disjoint from train at the training window (VERDICT CLEAN, 0 exact byte
# matches). The 32.72% measured during training came from the contaminated split and is NOT the number.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/HONEST-ACCURACY.txt
SHIP=$(ls -t lens/runs/chat/gn_d256_full_24_ep24.npz 2>/dev/null | head -1)
{
  echo "=== HONEST ACCURACY $(date '+%F %H:%M')"
  echo "ship: $SHIP"
  echo "val: data/chat/prepared_mix_honest (disjoint from train; VERDICT CLEAN by exact bytes)"
  echo "contaminated number measured during training: 32.72% (val had 48,615 overlapping windows)"
  echo
  EVAL_DATA_DIR="$PWD/data/chat/prepared_mix_honest" bash lens/scripts/run_eval_clean.sh "$SHIP"
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
grep -E "assistant-masked|plain pool|WARNING|VERDICT|acc" "$OUT" | tail -8