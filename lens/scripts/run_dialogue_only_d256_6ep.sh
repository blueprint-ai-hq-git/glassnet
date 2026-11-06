#!/bin/bash
# run_dialogue_only_d256_6ep.sh — THE DIALOGUE-ONLY ARM (the cheap half of option D).
# PREDICTIONS P110a-h WERE WRITTEN BEFORE THE BUILD SCRIPT, THE CORPUS AND THIS FILE EXISTED:
#   lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt
#
# ONE VARIABLE AGAINST THE CURRENT BEST HONEST RECIPE (lens/scripts/run_h2l_d256_6ep.sh, tag
# h2l_noconcept_169_6ep, best 37.2480% mixed / 24.8029% dialogue): CHAT_DATA_DIR points at
#   data/chat/prepared_dialogue_only_honest
# = conversations 0..156,519 of data/chat/prepared_mix_ctrl_honest, i.e. the ext_honest group (hh-rlhf, oasst1,
# ultrachat, orca-math = HUMAN DIALOGUE; 156,520 conversations, 38,996,999 tokens) instead of the whole mix.
# Every other flag is copied flag for flag: --d 256 --window 32 --epochs 6 --batch 256 --lr 0.4 --lr-decay 0.5
# --lam-concept 0 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode
# tied --max-windows 16900000 --method standard --ckpt-seconds 1800.
#
# THE CORPUS IS A SLICE, NOT A REBUILD, AND ITS CONTROLS PASSED BEFORE THIS LINE WAS QUEUED: same ids, same
# boundaries, same split carried over element for element, the mix's own vocab file copied byte for byte
# (sha256-compared), both control tokens present, 38,996,999 tokens = the exact count
# lens/runs/CORPUS-MIX-AUDIT.txt measured for the dialogue group. Evidence: lens/runs/DIALOGUE-ONLY-BUILD.txt
# (C1-C5 PASS). Wiring pre-flight: lens/runs/DIALOGUE-ONLY-PREFLIGHT.txt (dialogue_only_preflight.py).
#
# WHY: 85.57% of the assistant-turn tokens the loss trains on are maths/textbook/code/Gutenberg text
# [lens/runs/CORPUS-MIX-AUDIT.txt] while the ship criterion (lens/runs/PADFREE-TALK-*.txt) is entirely dialogue;
# the held-out column was split by population for the first time in lens/runs/DIALOGUE-VS-MIXED.txt (58,191
# assistant-by-position targets = 42,463 maths/code + 15,728 dialogue, and the dialogue column is flat across
# five checkpoints while the mixed column spans 2.10 points). Option D of the owner's decision file
# (lens/runs/DECISION-SHAPE-SCALE.md sections 4 and 7c) is "make the target population match the task" and this
# is its cheap half — never measured anywhere in this project. THE OWNER'S A/B/C/D CHOICE IS NOT MADE HERE: this
# arm measures ONE of the four, writes its own tag and its own log, overwrites nothing and deletes nothing.
#
# IT IS NOT EXPECTED TO TALK (P110d declares 0/6 in advance) and it is NOT a ship. The three deciders are
# chained in this same job so a shutdown cannot leave the arm unmeasured, and the fourth is the group-split
# instrument on the MIX corpus so this arm's dialogue column is measured by exactly the instrument that produced
# the stored 24.8029% / 23.5313%.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_dialogue_only_honest"
TAG=dial_only_d256_6ep
LOG=lens/runs/DIALOGUE-ONLY-D256-6EP.txt
{
  echo "=================================================================================================="
  echo "DIALOGUE-ONLY ARM — d=256, 6 epochs, the H2L recipe with CHAT_DATA_DIR = the human-dialogue slice"
  echo "  start $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (written before the corpus and this file)"
  echo "corpus:      $CHAT_DATA_DIR   (156,520 conversations, 38,996,999 tokens = 13.3% of the mix; SLICE of it)"
  echo "control:     lens/runs/DIALOGUE-ONLY-BUILD.txt (C1-C5 PASS: same ids, boundaries, split, vocab)"
  echo "P110a: the arm's own val_asst_n is predicted in [15,600, 15,728] (the dialogue column of"
  echo "       lens/runs/DIALOGUE-VS-MIXED.txt is exactly n=15,728; conv 156,519 is a TRAIN conversation so"
  echo "       15,728 is the expected value)"
  echo "P110b: dialogue-population accuracy above H2L's 24.8029% by +0.7..+8.2 pts (point estimate +2.7); a"
  echo "       value at or below 24.8029% CLOSES this option's cheap half and is reported as closed"
  echo "P110c: mixed-population accuracy 1-8 points BELOW H2L's 37.2480% (29.2-36.2%), the population signature"
  echo "P110d: THE DECIDER, declared in advance — 0 of 6 topical answers; any topical answer is the first talk"
  echo "       progress in ten arms and would be reported as such"
  echo "P110e: memorisation — 6.6 tokens/parameter against the twin's 2.86, so the ep6 train-minus-held gap is"
  echo "       predicted LARGER than H2L's +4.208 (in +5..+15)"
  echo "P110f: per-epoch wall 120-450 s, whole arm under 1.5 h, scope peak under 6 GB"
  echo "=================================================================================================="
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 6 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  LEGACY_BEST=$(.venv-gpu/bin/python3 -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/${TAG}_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print('%.4f'%float(b['val_masked_acc']))" 2>/dev/null)
  [ -z "$LEGACY_BEST" ] && LEGACY_BEST="0.0000"
  echo "this arm's best epoch read from its OWN jsonl, legacy by-token-id column as a FRACTION: $LEGACY_BEST"

  # ---- chained decider 1/3: the pad-free ship instrument on this arm's best + the twin ---------------------
  echo "--- chained decider 1/3: PAD-FREE SHIP INSTRUMENT on this arm's best AND the twin, ONE run (P110d) ---"
  CKPT_LIST="DLONLY arm (dialogue-only corpus - the H2L recipe with ONE variable changed)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the stored matched control)=lens/runs/chat/ct_d256_full169_fix3_best.npz"
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
    PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-DLONLY.txt" \
    taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
    echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  else
    echo "CKPT_LIST GUARD REFUSED - THE PAD-FREE INSTRUMENT WAS NOT STARTED (exit 3)  $(date '+%F %H:%M:%S')"
  fi

  # ---- chained decider 2/3: free-running by step on this arm's best ---------------------------------------
  echo "--- chained decider 2/3: FREE-RUNNING BY STEP on this arm's best, logged value read from its jsonl ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-DLONLY.txt" \
  TALK_FAIR_LOGGED="$LEGACY_BEST" \
  TALK_FAIR_EP="the arm's own best epoch (legacy by-id $LEGACY_BEST)" \
  TALK_FAIR_PREDICTIONS="lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (P110d)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  # ---- chained decider 3/3: the same instrument on the twin = the control for that number -----------------
  echo "--- chained decider 3/3: THE SAME INSTRUMENT ON THE TWIN (the control for that number) ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-DLONLY-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (P110d control)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-TWIN EXIT=$?  $(date '+%F %H:%M:%S')"

  # ---- decider 4/4 (the number the decision needs): the group split on the MIX corpus ---------------------
  # NOTE: decider 4 MUST run with the MIX corpus, not with the slice: the instrument reads the group boundary
  # from the corpus meta, and the slice holds one group only (meta_v12000.json says so in
  # DO_NOT_USE_PER_SOURCE_CORPUS_FOR_GROUP_SPLIT). The arm's OWN val column is its trainer's own line and is
  # already the dialogue column; this pass puts it in the same table as the stored 24.8029% / 37.2480%.
  echo "--- chained decider 4/4: DIALOGUE vs MIXED on the MIX corpus: this arm's best + H2L + the twin (P110b/c) ---"
  export CKPT_LIST="dlonly_d256_6ep=lens/runs/chat/${TAG}_best.npz;h2l_noconcept_6ep=lens/runs/chat/h2l_noconcept_169_6ep_best.npz;twin_d256_3ep=lens/runs/chat/ct_d256_full169_fix3_best.npz"
  bash lens/scripts/run_dialogue_mixed_eval.sh
  echo "DM EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ===  files: DIALOGUE-ONLY-D256-6EP.txt, PADFREE-TALK-DLONLY.txt,"
  echo "    TALK-FAIR-DLONLY.txt, TALK-FAIR-DLONLY-TWIN.txt, and the appended block in DIALOGUE-VS-MIXED.txt."
  echo "    P110b/P110c are the decision numbers; P110d is the only one that can make this a talk result."
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 6 "$LOG"
