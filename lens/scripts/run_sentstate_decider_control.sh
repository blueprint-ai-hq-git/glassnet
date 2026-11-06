#!/bin/bash
# run_sentstate_decider_control.sh -- P115e: does the ONE loader branch added to the ship decider
# (lens/scripts/talk_fair_v3_ship.py) leave every stored reading intact, take the grafted path when asked,
# and REFUSE a base checkpoint when the graft is asked for?
#
# Predictions: lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt (P115e), written before the branch existed.
# Output: lens/runs/SENTSTATE-DECODE.txt (append-only).
#
# PART 1 THE CONTROL -- the flag-ABSENT path must reproduce the stored pad-free block EXACTLY. The arm's own
#        block is already on disk (lens/runs/PADFREE-TALK-H2L.txt, mean pad-free <unk> 5.6% over six greedy
#        24-token samples). Greedy decode is deterministic, so ANY difference in a single token means the edit
#        changed the instrument. The new run writes a NEW file; the stored file is never touched.
# PART 2 THE BRANCH -- flag ON, tiny corpus + the pre-flight's real GRAFTED checkpoint: the decider must build
#        the net through load_grafted() and print its blocks. Tiny vocab: NO quality claim, wiring only.
# PART 3 THE REFUSAL -- flag ON with a BASE checkpoint must FAIL BY NAME, not silently drop six tensors.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=.venv-gpu/bin/python3
OUT=lens/runs/SENTSTATE-DECODE.txt
RUN="taskset -c 2,3 nice -n 19 ionice -c3 $PY -u lens/scripts/talk_fair_v3_ship.py"
STAMP="$(date '+%F %H:%M:%S')"

{
  echo ""
  echo "####################################################################################################"
  echo "# P115e  THE DECIDER'S ONE LOADER BRANCH -- control, branch, refusal   $STAMP"
  echo "# predictions: lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt   instrument: lens/scripts/talk_fair_v3_ship.py"
  echo "####################################################################################################"

  echo ""
  echo "=== PART 1 CONTROL: flag ABSENT on the stored base checkpoint h2l_noconcept_169_6ep_best.npz"
  echo "    must reproduce lens/runs/PADFREE-TALK-H2L.txt (mean pad-free <unk> 5.6%, six greedy strings)"
  CKPT_LIST="H2L base (flag OFF control) = lens/runs/chat/h2l_noconcept_169_6ep_best.npz" \
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-H2L-CTRL-P115.txt" \
    $RUN
  echo "PART1 EXIT=$?  $(date '+%H:%M:%S')"

  echo ""
  echo "=== PART 2 BRANCH: GLASSNET_SENTSTATE=1 on the REAL grafted pre-flight checkpoint (tiny corpus)"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small" \
  GLASSNET_SENTSTATE=1 \
  CKPT_LIST="grafted tiny preflight (flag ON) = lens/runs/_tmp/ss_preflight/_ss_wire_on_ep1.npz" \
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SENTSTATE-TINY.txt" \
    $RUN
  echo "PART2 EXIT=$?  $(date '+%H:%M:%S')"

  echo ""
  echo "=== PART 3 REFUSAL: GLASSNET_SENTSTATE=1 on a BASE checkpoint -- must fail BY NAME"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small" \
  GLASSNET_SENTSTATE=1 \
  CKPT_LIST="base tiny (must be refused) = lens/runs/_tmp/ss_preflight/_ss_wire_off_ep1.npz" \
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SENTSTATE-REFUSAL.txt" \
    $RUN
  echo "PART3 EXIT=$?  (NOT 0 is the PASS condition for this part)  $(date '+%H:%M:%S')"

  echo ""
  echo "=== COMPARISON $(date '+%H:%M:%S')"
  nice -n 19 $PY -u lens/scripts/sentstate_decider_control_check.py
  echo "CHECK EXIT=$?"
  echo "=== P115e DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1

sed -n '/P115e  THE DECIDER/,$p' "$OUT" | tail -30
