#!/bin/bash
# Wait for the combined corpus, then run the steps that were skipped because it was missing.
#
# WHY: LADDER-BIG.txt says "## ABORT: combined corpus missing" and MIX-SMALL.txt is empty. The master run
# had already passed those steps in its order, so nothing would ever retry them -- the work would silently
# never happen. That is exactly the kind of gap the user said to fix instead of reporting.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/CORPUS-THEN-REST.txt
{
  echo "=== CORPUS-THEN-REST $(date '+%F %H:%M')"
  for i in $(seq 1 180); do            # up to 3 hours of waiting, checked every minute
    if [ -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ]; then
      echo "corpus present after $i minutes: $(date '+%H:%M')"
      du -sh data/chat/prepared_mix_ctrl_honest
      break
    fi
    sleep 60
  done
  if [ ! -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ]; then
    echo "## ABORT: corpus still missing after 3 hours. The corpus build is the blocker; nothing downstream"
    echo "## can run without it. Reported here rather than looping forever."
    exit 1
  fi
  echo "--- small smart model on the combined corpus"
  bash lens/scripts/run_mix_small.sh
  echo "--- big rungs (d=1536/2048/3072)"
  bash lens/scripts/run_ladder_big.sh
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
