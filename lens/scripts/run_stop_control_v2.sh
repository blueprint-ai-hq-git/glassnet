#!/bin/bash
# The stop gate through the shared decoder, twice: with the transparent rules OFF (what the model does
# on its own) and ON (what the product does). Reporting both is the point -- the difference between them
# is exactly how much stopping is learned versus rule-forced.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"
SHIP="${1:-lens/runs/chat/te_T1_turnend_d256_24_ep24.npz}"
OUT=lens/runs/STOP-V2.txt
{
  echo "=== STOP CONTROL v2 $(date '+%F %H:%M')  ship $(basename "$SHIP")"
  echo
  echo "--- A: rules OFF (what the model does by itself)"
  SHIP="$SHIP" RULE_STOP=0 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/stop_control_v2.py
  echo
  echo "--- B: rules ON (what the product does)"
  SHIP="$SHIP" RULE_STOP=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/stop_control_v2.py
} 2>&1 | tee "$OUT"
echo "exit=$?"