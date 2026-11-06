#!/bin/bash
# run_sentstate_arm.sh -- THE SENTSTATE ARM: the matched twin's recipe PLUS the running sentence state graft.
# Predictions P116a-h were written BEFORE this file existed: lens/runs/SENTSTATE-ARM-PREDICTIONS-P116.txt.
#
# WHY THIS ARM AND NOT ANOTHER FLAG ON THE SAME MODEL: the answer path has never had a function of ORDER over
# the window beyond dmix (lens/model.py: logits = answer(sigmoid(concept(x))) + direct path). The graft adds a
# carried state s_t = tanh(hp_t @ Wc + s_{t-1} @ Ur + b) over the 32 positions and a parallel readout of it
# (lens/sentstate_wire.py). Everything else -- corpus, seed, batch, lr, mask, tied direct path, epochs -- is
# the stored twin's recipe, flag for flag, so the arm is ONE variable.
#
# WHAT IS ALREADY PROVEN (lens/runs/SENTSTATE-WIRE.txt P114, lens/runs/SENTSTATE-DECODE.txt P115):
#   * flag-absent path field-identical to the unmodified trainer (19/19 epoch-line fields, P114a);
#   * all six graft tensors carry nonzero gradients; the grafted checkpoint reloads bit-exactly (0.0);
#   * the KV cache was measured STATE-BLIND on a grafted net (2.510019 logit difference, top-1 0/8, P115b)
#     and the new grafted cached step reproduces the full path exactly (P115c, 8/8 top-1);
#   * the ship decider loads a grafted checkpoint through load_grafted() and its flag-absent reading is
#     reproduced to the character (P115e: 6/6 greedy strings, mean <unk> 5.6%).
# So this arm is the FIRST one whose deciders are guaranteed to be measuring the model that was trained.
#
# THE CONTROL IS THE STORED TWIN (read-only): ct_d256_full169_fix3 -> 32.5222 / 34.6170 / 35.2632% by
# position, gap +2.244/+4.042/+3.308, pad-free <unk> 2.1%, step-8 free-running 3.9%, 0/6 answers.
# DECLARED IN ADVANCE (P116d): 0 of 6 pad-free prompts are expected to be answered. This is NOT expected to
# talk; if it talks, that changes the project and gets its own grade before anyone is told.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
TAG=ss_d256_full169_3ep
TWIN=ct_d256_full169_fix3
LOG=lens/runs/SENTSTATE-ARM.txt
PY=.venv-gpu/bin/python3
{
  echo "================================================================================================"
  echo "SENTSTATE ARM (the twin's recipe + the running sentence state)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SENTSTATE-ARM-PREDICTIONS-P116.txt (written before this runner existed)"
  echo "recipe = ct_d256_full169_fix3 flag for flag; the ONLY variable is the graft (SENTSTATE=1, ds=64)"
  echo "P116a: the model line must read 6,704,496 parameters = 5,908,686 + 795,810. Nothing else is admissible."
  echo "twin (stored, read-only): 32.5222 / 34.6170 / 35.2632% by position; drift floor 0.0739 points"
  echo "================================================================================================"

  SENTSTATE=1 SENTSTATE_DS=64 $PY -u lens/scripts/train_chat_sent.py \
    --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  BEST=$($PY -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/${TAG}_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print('%s %.4f %.4f'%(b.get('epoch'),float(b['val_asst_acc'])*100,float(b['val_masked_acc'])))" 2>/dev/null)
  echo "BEST EPOCH (by position, then legacy by-id): $BEST   <- the deciders below read THIS checkpoint"
  LEGACY_BEST=$(echo "$BEST" | awk '{print $3}')
  [ -z "$LEGACY_BEST" ] && LEGACY_BEST="0.0000"
  if [ ! -f "lens/runs/chat/${TAG}_best.npz" ]; then
    echo "NO BEST CHECKPOINT - the arm wrote nothing usable; deciders are NOT run on the last checkpoint."
    echo "=== DONE $(date '+%F %H:%M:%S') ==="
    exit 1
  fi

  # ---------------- decider 1/2: THE SHIP DECIDER (pad-free), arm then twin, two runs because the twin is a
  # BASE checkpoint and the grafted loader REFUSES a base file by name (that refusal is P115e(iii)).
  echo ""
  echo "--- decider 1a: PAD-FREE SHIP INSTRUMENT on the arm's best checkpoint, GLASSNET_SENTSTATE=1  (P116d) ---"
  CKPT_LIST="SENTSTATE arm = lens/runs/chat/${TAG}_best.npz"
  export CKPT_LIST
  if $PY -c "
import os,sys
items=os.environ['CKPT_LIST'].split(';')
bad=[i for i in items if '=' not in i]
print('CKPT_LIST guard: %d item(s)'%len(items))
if bad:
    print('REFUSING: item(s) with no = : %r'%bad); sys.exit(3)
print('CKPT_LIST guard: PASS')
"; then
    GLASSNET_SENTSTATE=1 PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SENTSTATE.txt" \
      taskset -c 2,3 nice -n 19 ionice -c3 $PY -u lens/scripts/talk_fair_v3_ship.py
    echo "PADFREE(arm) EXIT=$?  $(date '+%H:%M:%S')"
  else
    echo "CKPT_LIST GUARD REFUSED - PAD-FREE INSTRUMENT NOT STARTED  $(date '+%H:%M:%S')"
  fi

  echo ""
  echo "--- decider 1b: PAD-FREE SHIP INSTRUMENT on the STORED TWIN (flag unset - the matched control) ---"
  CKPT_LIST="TWIN ${TWIN} (stored, matched control) = lens/runs/chat/${TWIN}_best.npz"
  export CKPT_LIST
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SENTSTATE-TWIN.txt" \
    taskset -c 2,3 nice -n 19 ionice -c3 $PY -u lens/scripts/talk_fair_v3_ship.py
  echo "PADFREE(twin) EXIT=$?  $(date '+%H:%M:%S')"

  # ---------------- decider 2/2: FREE-RUNNING BY STEP (the lever's own number, P116e)
  echo ""
  echo "--- decider 2a: FREE-RUNNING BY STEP on the arm's best, GLASSNET_SENTSTATE=1  (P116e) ---"
  GLASSNET_SENTSTATE=1 \
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-SENTSTATE.txt" \
  TALK_FAIR_LOGGED="$LEGACY_BEST" \
  TALK_FAIR_EP="the arm's own best epoch (legacy by-id $LEGACY_BEST)" \
  TALK_FAIR_PREDICTIONS="lens/runs/SENTSTATE-ARM-PREDICTIONS-P116.txt (P116e)" \
    taskset -c 2,3 nice -n 19 ionice -c3 $PY -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR(arm) EXIT=$?  $(date '+%H:%M:%S')"

  echo ""
  echo "--- decider 2b: FREE-RUNNING BY STEP on the STORED TWIN (flag unset - the same-run control) ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TWIN}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-SENTSTATE-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/SENTSTATE-ARM-PREDICTIONS-P116.txt (P116e control)" \
    taskset -c 2,3 nice -n 19 ionice -c3 $PY -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR(twin) EXIT=$?  $(date '+%H:%M:%S')"

  echo ""
  echo "=== DONE $(date '+%F %H:%M:%S') ===  files: SENTSTATE-ARM.txt, PADFREE-TALK-SENTSTATE.txt,"
  echo "    PADFREE-TALK-SENTSTATE-TWIN.txt, TALK-FAIR-SENTSTATE.txt, TALK-FAIR-SENTSTATE-TWIN.txt,"
  echo "    chat/${TAG}_history.jsonl, chat/${TAG}_best.npz.  NOTHING SHIPS ON ANY NUMBER IN THESE FILES:"
  echo "    the verdict is P116d, read by eye from the pad-free blocks, and it is declared 0/6 in advance."
} >> "$LOG" 2>&1
