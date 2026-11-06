#!/bin/bash
# SHIP A MODEL: export it in both standard containers, verify them, and record it.
# This is the step that must run for EVERY model we ship, so it is one command:
#   bash lens/scripts/run_ship.sh <ship.npz>
# It writes .safetensors and .gguf (plus the model card and the native .npz) into lens/runs/export/,
# verifies every file by reading it back, and appends a line to SHIPS.md with the honest numbers that
# are known for that ship. Nothing is claimed here that was not measured: unmeasured fields say so.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP="${1:?usage: run_ship.sh <ship.npz>}"
[ -f "$SHIP" ] || { echo "no such ship: $SHIP"; exit 1; }
BASE=$(basename "$SHIP" .npz)
LOG=/tmp/ship_${BASE}.log

nice -n 19 ionice -c3 /home/yahya/hermes-env/bin/python3 -u lens/export.py "$SHIP" lens/runs/export > "$LOG" 2>&1
RC=$?
tail -22 "$LOG"
if [ "$RC" -ne 0 ]; then
  echo "EXPORT FAILED for $SHIP -- nothing recorded in SHIPS.md"
  exit 1
fi

# honest numbers we already have for this ship, if any (never invented)
STOP_JSON="lens/runs/stop_control_${BASE}.json"
READ_JSON="lens/runs/readability_random_${BASE}.json"
STOP_TXT="not measured"
READ_TXT="not measured"
[ -f "$STOP_JSON" ] && STOP_TXT=$(python3 -c "
import json;d=json.load(open('$STOP_JSON'));print(f\"handed back {d['handed_back']}/{d['n']}, unique tokens {d['mean_unique_ratio']*100:.1f}%\")")
[ -f "$READ_JSON" ] && READ_TXT=$(python3 -c "
import json;d=json.load(open('$READ_JSON'));print(f\"verdict {d['verdict']}, false fires {len(d['false_fires'])}, explain residual {d['explain_residual_max']:.1e}\")")

{
  echo "## $BASE"
  echo "- files: \`lens/runs/export/$BASE.safetensors\`, \`lens/runs/export/$BASE.gguf\`, \`lens/runs/export/$BASE.npz\`, \`lens/runs/export/$BASE.model-card.md\`"
  echo "- exported: $(date '+%F %H:%M'), all files verified by reading back"
  echo "- stop control: $STOP_TXT"
  echo "- random readability: $READ_TXT"
  echo "- note: standard containers, but this architecture needs our loader; llama.cpp / LM Studio cannot run it"
  echo
} >> SHIPS.md
echo "recorded in SHIPS.md"