#!/bin/bash
# THE CURRENT BEST HONEST RECIPE, TRAINED LONGER — predictions P104a-h, written BEFORE this file existed:
# lens/runs/H2L-PREDICTIONS-P104.txt.
#
# ONE FLAG FROM THE H2 ARM (lens/scripts/run_h2_noconcept.sh): --epochs 3 -> 6. Everything else is copied flag
# for flag, including --lam-concept 0 (the readability-cost ablation's flag) and the twin's other flags from
# lens/scripts/run_tied_full169_maskfix.sh:
#   --d 256 --window 32 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0 --lam-act 0.0 --seed 0 --device gpu
#   --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 16900000
#   --method standard --ckpt-seconds 1800 (explicit defaults + the verified 30-minute mid-epoch checkpoint).
#
# WHY THIS ARM, from files graded this turn (lens/runs/GRADE-P102.txt): H2 is the best honest ship candidate at
# d=256 — it beat the stored twin at ALL three epochs (33.0000 / 35.0432 / 35.7220 by position, n=58,191, against
# 32.5222 / 34.6170 / 35.2632), with a SMALLER train-vs-held gap (+1.864 / +3.883 / +3.204 against
# +2.244 / +4.042 / +3.308) and a held-out perplexity still falling at ep3 (31.668 against the twin's 32.771).
# It was still improving when its last epoch ended, so the ship's number has room here — and the six-epoch
# concept-ON arm (bash lens/scripts/run_h5l_d256_6ep.sh, tag ct_d256_full169_6ep, started by the watchdog at
# 15:42:38 into lens/runs/TIED-D256-6EP.txt) is the other half of a clean 2x2: concept supervision ON/OFF at
# BOTH 3 and 6 epochs, same corpus, seed, batch, window order, one flag apart.
#
# IT IS NOT EXPECTED TO TALK (P104g says 0/6 in advance). Every cheap lever on the talk path is closed with its
# number (decoder P73/P74, readout P77, attention P82, <unk> P89, window 64 P95, exposure bias P98, concept
# supervision P102). This arm is a ship-candidate refinement and a memorisation measurement, NOT talk progress.
#
# P104b, WRITTEN CORRECTLY THIS TIME: ep1 (lr 0.4, same seed/corpus/batch/window order) is the STRICT
# reproduction control and must reproduce H2's own ep1 33.0000% to within 0.01 points. ep2/ep3 are lr-CONFOUNDED
# (this arm prints 0.3600/0.3200 where H2 printed 0.3/0.2) and carry no pass/fail — see BUGS.md
# [BUG-PREDICTION-PROMISES-A-CONTROL-THAT-CANNOT-HOLD], which is why P102c's wording is not reused here.
#
# THE DECIDERS ARE CHAINED IN THIS SAME JOB so a shutdown cannot leave the arm unmeasured (same three
# instruments and the same repaired environment as run_h2_noconcept.sh): no ';' in any CKPT_LIST label
# (the caller-side guard refuses with exit 3 before the instrument starts), and TALK_FAIR_LOGGED read from THIS
# arm's OWN jsonl as a FRACTION instead of guessed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
TAG=h2l_noconcept_169_6ep
LOG=lens/runs/H2L-D256-6EP.txt
{
  echo "=================================================================================================="
  echo "H2L — THE CURRENT BEST RECIPE (concept supervision OFF), SIX EPOCHS AT d=256  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/H2L-PREDICTIONS-P104.txt (written before this file existed)"
  echo "recipe = the H2 arm PLUS --epochs 6 (one flag); H2 = the twin PLUS --lam-concept 0"
  echo "strict control (P104b): ep1 by position reproduces H2's own ep1 33.0000% to within 0.01 points"
  echo "P104c: ep6 by position predicted 36.0-38.0%; FALSIFIED at or below 35.7220%"
  echo "P104d: gap under +6 points at ep6, no peak-then-decline before it"
  echo "P104e: ep6 at or above the concept-ON six-epoch arm (ct_d256_full169_6ep); falsified below it by >1 point"
  echo "P104g: predicted 0/6 topical answers — NOT talk progress, by this project's own rule"
  echo "=================================================================================================="
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 6 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  LEGACY_BEST=$(.venv-gpu/bin/python3 -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/${TAG}_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print('%.4f'%float(b['val_masked_acc']))" 2>/dev/null)
  [ -z "$LEGACY_BEST" ] && LEGACY_BEST="0.0000"
  echo "this arm's best epoch, legacy by-token-id column as a FRACTION for the instrument: $LEGACY_BEST"

  echo "--- chained decider 1/3: PAD-FREE SHIP INSTRUMENT on this arm's best AND the twin, ONE run (P104g) ---"
  CKPT_LIST="H2L arm (concept OFF - the H2 recipe with 6 epochs)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the matched control - stored - concept supervision ON, 3 epochs)=lens/runs/chat/ct_d256_full169_fix3_best.npz"
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
    PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-H2L.txt" \
    taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
    echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  else
    echo "CKPT_LIST GUARD REFUSED - THE PAD-FREE INSTRUMENT WAS NOT STARTED (exit 3)  $(date '+%F %H:%M:%S')"
  fi

  echo "--- chained decider 2/3: FREE-RUNNING BY STEP on this arm's best, logged value READ FROM ITS OWN jsonl ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-H2L.txt" \
  TALK_FAIR_LOGGED="$LEGACY_BEST" \
  TALK_FAIR_EP="the arm's own best epoch (legacy by-id $LEGACY_BEST)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2L-PREDICTIONS-P104.txt (P104g)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-ARM EXIT=$?  $(date '+%F %H:%M:%S')"

  echo "--- chained decider 3/3: THE SAME INSTRUMENT ON THE TWIN = the control for that number ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-H2L-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/H2L-PREDICTIONS-P104.txt (P104g control)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-TWIN EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ===  files: H2L-D256-6EP.txt, PADFREE-TALK-H2L.txt, TALK-FAIR-H2L.txt, TALK-FAIR-H2L-TWIN.txt. P104c/P104d/P104e/P104g decide whether this arm is the ship's best number; NONE of them is a talk claim."
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 6 "$LOG"
