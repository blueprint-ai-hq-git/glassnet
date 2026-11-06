#!/bin/bash
# ARM 2 OF THIS TURN — H5-STYLE LONGER SCHEDULE AT THE WIDTH THAT DOES NOT OVERFIT (predictions P102a-h, written
# BEFORE this file existed: lens/runs/H2-H5L-PREDICTIONS-P102.txt).
#
# ONE FLAG FROM THE TWIN: --epochs 3 -> 6. Every other flag is copied from lens/scripts/run_tied_full169_maskfix.sh
# (the runner that produced ct_d256_full169_fix3), plus --method standard --ckpt-seconds 1800.
# DECLARED CONFOUND, same as the d=512 ext6 arm: with --lr-decay 0.5 the anneal stretches over 6 epochs, so
# ep2-ep6 do not have the twin's lr schedule; ep1 is the near-reproduction and ep1-3 are the reproduction control.
#
# WHY THIS ARM: the project's own next-candidate list (STATE.md 11:4x item 7(c)(ii)) names H2 and an H5-style
# longer/wider schedule. The d=512 capacity test is CLOSED (+0.35 for 2.6x params, P69; +0.19 for three more
# epochs, P94c) and the d=256 twin is the width that does NOT overfit (train-vs-held gap +3.31 points at ep3,
# against +12.89 for d=512/6ep), so "longer at d=256" is the version of the schedule question that still has room.
# It also gives the plan's item 3 (the memorisation attack) its first target at more than three epochs on the
# honest mask at this width: P102f predicts the gap stays under +6 points at ep6 with no peak before it.
#
# P102c IS THE STRONGEST CONTROL IN THIS TURN: because this is the twin's run with the same seed, corpus, batch and
# window order, its ep1/ep2/ep3 rows must reproduce the twin's 32.5222% / 34.6170% / 35.2632% by position
# (n=58,191) to within 0.01 points. That is also the full-scale check that the trainer's DEFAULT path still
# reproduces the stored twin after the 2026-10-25 self-context edits (P98a proved it only at d=64).
#
# THE DECIDERS ARE CHAINED IN THIS SAME JOB (same three instruments, same repaired env as run_h2_noconcept.sh:
# no ';' in any CKPT_LIST label, and TALK_FAIR_LOGGED read from this arm's OWN jsonl as a FRACTION).
# THIS ARM IS NOT EXPECTED TO TALK EITHER (P102g says 0/6 in advance).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
TAG=ct_d256_full169_6ep
LOG=lens/runs/TIED-D256-6EP.txt
{
  echo "================================================================================================"
  echo "H5-STYLE LONGER SCHEDULE AT d=256 (the twin's recipe, 6 epochs)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/H2-H5L-PREDICTIONS-P102.txt (written before this file existed)"
  echo "recipe = the twin ct_d256_full169_fix3 PLUS --epochs 6 (one flag changed)"
  echo "twin (stored, 3 epochs): by position 32.5222 / 34.6170 / 35.2632, train 38.57%, gap +3.31, step-8 3.9%,"
  echo "                         pad-free <unk> 2.1%, 0/6 answers"
  echo "P102c CONTROL: ep1-3 of THIS arm must reproduce 32.5222 / 34.6170 / 35.2632 to within 0.01 points."
  echo "P102e: ep6 by position predicted 35.8-37.5%; falsified at or below 35.2632%."
  echo "================================================================================================"
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 6 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  LEGACY_BEST=$(.venv-gpu/bin/python3 -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/${TAG}_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print('%.4f'%float(b['val_masked_acc']))" 2>/dev/null)
  [ -z "$LEGACY_BEST" ] && LEGACY_BEST="0.0000"
  echo "this arm's best epoch, legacy by-token-id column as a FRACTION for the instrument: $LEGACY_BEST"

  echo "--- chained decider 1/3: PAD-FREE SHIP INSTRUMENT on this arm's best AND the twin, ONE run (P102g) ---"
  # PRE-FLIGHT GUARD; see the note in run_h2_noconcept.sh and [BUG-CKPTLIST-LABEL-SEMICOLON].
  CKPT_LIST="d256 6-epoch arm (the twin's recipe with the schedule stretched to 6 epochs)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the matched 3-epoch control - stored)=lens/runs/chat/ct_d256_full169_fix3_best.npz"
  export CKPT_LIST
  if .venv-gpu/bin/python3 -c "
import os,sys
items=os.environ['CKPT_LIST'].split(';')
bad=[i for i in items if '=' not in i]
print('CKPT_LIST guard: %d item(s)'%len(items))
if bad:
    print('REFUSING: item(s) with no = (a label may not contain the ; separator): %r'%bad)
    sys.exit(3)
print('CKPT_LIST guard: PASS - every item carries a path')
"; then
    PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-D256-6EP.txt" \
    taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
    echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  else
    echo "CKPT_LIST GUARD REFUSED - THE PAD-FREE INSTRUMENT WAS NOT STARTED (exit 3)  $(date '+%F %H:%M:%S')"
  fi

  echo "--- chained decider 2/3: FREE-RUNNING BY STEP on this arm's best, logged value READ FROM ITS OWN jsonl ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-D256-6EP.txt" \
  TALK_FAIR_LOGGED="$LEGACY_BEST" \
  TALK_FAIR_EP="the arm's own best epoch (legacy by-id $LEGACY_BEST)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2-H5L-PREDICTIONS-P102.txt (P102g)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  echo "--- chained decider 3/3: THE SAME INSTRUMENT ON THE TWIN = the control for that number ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-D256-6EP-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2-H5L-PREDICTIONS-P102.txt (P102g control)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-TWIN EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 6 "$LOG"
