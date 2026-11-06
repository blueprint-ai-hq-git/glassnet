#!/bin/bash
# ARM 1 OF THIS TURN — H2, THE READABILITY-COST ABLATION (predictions P102a-g, written BEFORE this file existed:
# lens/runs/H2-H5L-PREDICTIONS-P102.txt).
#
# ONE FLAG FROM THE TWIN: --lam-concept 0. Everything else is the twin's exact recipe, copied flag for flag from
# lens/scripts/run_tied_full169_maskfix.sh (the runner that produced ct_d256_full169_fix3):
#   --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-act 0.0 --seed 0 --device gpu
#   --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 16900000
# Only addition besides the one flag: --method standard --ckpt-seconds 1800 (explicit defaults + the verified
# 30-minute mid-epoch checkpoint, P81a-f), which the twin's runner left implicit and the self-context runner also
# passed.
#
# WHAT IT TESTS, in the owner's own words (GOAL.md section 3): "A3 measured readable as slightly worse than the
# plain normal on both accuracy and perplexity. So 'smarter' is not established." NEXT-EXPERIMENTS.md H2 is the
# cheap version of that question and has never been run at full-corpus scale. At --lam-concept 0 the 36-concept
# auxiliary loss is skipped (lens/model.py line 319 gates it on self.lam_concept > 0) while the concept layer
# STAYS in the architecture and still feeds the readout through answer.W, so this isolates the SUPERVISION.
#
# THE TWIN IS THE CONTROL AND IS ALREADY ON DISK: ct_d256_full169_fix3, ep1/ep2/ep3 by position 32.5222% /
# 34.6170% / 35.2632% (n=58,191), train 38.57%, concept loss 0.4405 / 0.1141 / 0.0752, step-8 free-running 3.9%,
# pad-free <unk> 2.1%, 0/6 answers. This arm's own concept column must read 0.0000 — that is P102b's control.
#
# THE DECIDERS ARE CHAINED IN THIS SAME JOB so a shutdown cannot leave the arm unmeasured:
#   * the pad-free ship instrument (the DECIDER) on this arm's best AND the twin, ONE run — labels contain no ';'
#     (that is [BUG-CKPTLIST-LABEL-SEMICOLON], which killed the self-context job's chained decider).
#   * free-running by step on this arm's best, with TALK_FAIR_LOGGED READ FROM THIS ARM'S OWN jsonl (a FRACTION;
#     passing a guessed value is what made the self-context job's control print a false FAIL), and the same
#     instrument on the twin as the instrument's control.
#
# THIS ARM IS NOT EXPECTED TO TALK (P102g says so in advance, 0/6). It is the project's own next candidate (ii).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
TAG=h2_noconcept_169_3
LOG=lens/runs/H2-NOCONCEPT.txt
{
  echo "================================================================================================"
  echo "H2 — THE READABILITY-COST ABLATION (lam_concept=0)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/H2-H5L-PREDICTIONS-P102.txt (written before this file existed)"
  echo "recipe = the twin ct_d256_full169_fix3 PLUS --lam-concept 0 (one flag)"
  echo "twin to beat (stored): ep3 by position 35.2632%, train 38.57%, concept loss 0.4405/0.1141/0.0752,"
  echo "                       step-8 free-running 3.9%, pad-free <unk> 2.1%, 0/6 answers"
  echo "my expectation: a small GAIN (0 to +3 points) or at most -2; falsifier at or below 33.2632%"
  echo "================================================================================================"
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  LEGACY_BEST=$(.venv-gpu/bin/python3 -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/${TAG}_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print('%.4f'%float(b['val_masked_acc']))" 2>/dev/null)
  [ -z "$LEGACY_BEST" ] && LEGACY_BEST="0.0000"
  echo "this arm's best epoch, legacy by-token-id column as a FRACTION for the instrument: $LEGACY_BEST"

  echo "--- chained decider 1/3: PAD-FREE SHIP INSTRUMENT on this arm's best AND the twin, ONE run (P102g) ---"
  # PRE-FLIGHT GUARD (added this turn, after [BUG-CKPTLIST-LABEL-SEMICOLON] fired a SECOND time — inside the
  # self-context job's chained decider, where it cost the ship decider its only run): ';' is the instrument's own
  # item separator, so a LABEL containing one silently becomes two items and the second has no '='. This refuses
  # LOUDLY before the instrument is started, and it needs no corpus and no GPU.
  CKPT_LIST="H2 arm (lam_concept 0 - the twin's recipe with the concept supervision OFF)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the matched control - stored - concept supervision ON)=lens/runs/chat/ct_d256_full169_fix3_best.npz"
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
    PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-H2.txt" \
    taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
    echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  else
    echo "CKPT_LIST GUARD REFUSED - THE PAD-FREE INSTRUMENT WAS NOT STARTED (exit 3)  $(date '+%F %H:%M:%S')"
  fi

  echo "--- chained decider 2/3: FREE-RUNNING BY STEP on this arm's best, logged value READ FROM ITS OWN jsonl ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-H2.txt" \
  TALK_FAIR_LOGGED="$LEGACY_BEST" \
  TALK_FAIR_EP="the arm's own best epoch (legacy by-id $LEGACY_BEST)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2-H5L-PREDICTIONS-P102.txt (P102g)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  echo "--- chained decider 3/3: THE SAME INSTRUMENT ON THE TWIN = the control for that number ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-H2-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2-H5L-PREDICTIONS-P102.txt (P102g control)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-TWIN EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 6 "$LOG"
