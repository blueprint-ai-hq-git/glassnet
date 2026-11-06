#!/bin/bash
# Re-run the leakage meter (fixed: own-label mask is per-window, not per-position).
# Target: our current best ship, on its own corpus, so the pairing is correct.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec > /tmp/leakage_ship.log 2>&1
echo "=== LEAKAGE RERUN $(date '+%F %H:%M')"
bash lens/scripts/run_concept_leakage.sh lens/runs/chat/t12_splitfix_d512_ep192.npz \
  "$PWD/data/chat/prepared_splitfix" 20000
echo "=== LEAKAGE RERUN DONE $(date '+%F %H:%M')"