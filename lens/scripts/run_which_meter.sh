#!/bin/bash
# WHICH METER IS WRONG? The recorded ship is supposed to score 63.85% (assistant-masked).
# Run the PROJECT's own eval meter on it twice: on the curated corpus (where 63.85% was recorded) and
# on the splitfix corpus. Then compare against my cross_eval's 17.88% on the same splitfix text.
# A control that cannot reproduce a recorded number invalidates every number this session produced
# through that meter, so this decides what gets quoted and what gets thrown away.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP=lens/runs/chat/t12_splitfix_d512_ep192.npz
echo "=== (a) project meter, DEFAULT corpus (prepared_clean) -- the recorded 63.85% setup"
bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 | grep -E "SHIP|assistant-masked|plain pool|windows|targets"
echo
echo "=== (b) project meter, EVAL_DATA_DIR=prepared_splitfix (the corpus the ship trained on)"
EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
  | grep -E "SHIP|assistant-masked|plain pool|windows|targets"
echo
echo "=== (c) my cross_eval on prepared_splitfix, for comparison: 17.88% (from /tmp/cross_eval.log)"