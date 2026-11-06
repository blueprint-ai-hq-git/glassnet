#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# preserve yesterday's (plain-parsed) big rung artifact — never overwrite old evidence
cp lens/runs/ladder/ladder_big.json lens/runs/ladder/ladder_big_prior_plainparsed.json 2>/dev/null
# BIG RUNG RE-VERIFICATION after the wrong-subject bug fix:
# yesterday's ladder-big image_cls 86.42% was parsed from the LAST TEST line = the PLAIN control.
# This re-runs the full big rung (24 epochs, 60k, GPU, all 7 capabilities) with readable_test().
# PRE-REGISTERED: p1 readable > 2x chance (gate) | p2 plain-vs-readable delta stays small at big
# scale like it was at small (65.76/65.81) | p3 the OTHER six arms repeat their old PASS (pure
# regression — nothing else changed, so anything else failing is a real new break).
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/ladder.py --rung big &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
