#!/bin/bash
# SELF-CONTEXT / EXPOSURE-BIAS ARM — the one ship-path axis this project has never trained.
# Predictions P98a-h: lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt (written BEFORE this runner existed; the
# wiring pre-flight is lens/runs/SELF-CONTEXT-PREFLIGHT.txt and its verdicts are P98a-c).
#
# WHY THIS LINE AND NOT SOMETHING ELSE, from files only: the free-running collapse is the measured failure that
# remains (step-8 free-running 3.9% vs teacher-forced 23.0% [lens/runs/TALK-FAIR-finalbest-d256-repro.txt];
# 1.6% vs 22.3% for window 64 [lens/runs/TALK-FAIR-W64.txt]; 2.3% vs 20.7% for the <unk>-drop arm
# [lens/runs/TALK-FAIR-UNKDROP.txt]) and every lever that assumed the fault was CONTEXT or READOUT is closed by
# measurement: window 64 (P95c/d), attention (P82), decoder (P73/P74), readout (P77), the <unk> objective (P89).
# Training has always fed REAL corpus tokens as context; this is the first arm that asks the model to predict
# real English from a context that contains its OWN errors.
#
# IT IS THE TWIN'S EXACT RECIPE, ONE THING CHANGED: ct_d256_full169_fix3 (16.9M windows, d=256, W=32, batch 256,
# lr 0.4, lr-decay 0.5, seed 0, tied direct path, assistant-only by position, train-turn-end, 3 epochs) plus
# --self-context-frac 0.5 --self-context-rounds 4. The twin's stored numbers are the comparison:
#   ep1 32.52% / ep2 34.62% / ep3 35.26% by position, train 38.57%, gap +3.31 points, step-8 free-running 3.9%,
#   pad-free talk 2.1% <unk> with 0/6 topical answers [lens/runs/chat/ct_d256_full169_fix3_history.jsonl,
#   lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt, lens/runs/TALK-FAIR-finalbest-d256-repro.txt,
#   lens/runs/PADFREE-TALK-ATTN.txt].
#
# THE DECIDER IS CHAINED IN THIS SAME JOB so a shutdown cannot leave the arm unmeasured:
#   * PAD-FREE TALK (P98f, the ship decider): the v3 instrument on the arm's best AND the twin, ONE run.
#   * FREE-RUNNING BY STEP (P98e, the lever's own falsifier): talk_fair_test.py on the arm's best, and on the
#     twin as the instrument's control (must reproduce the stored 3.9%).
# Own log, own --out tag, own instruments' outputs: nothing stored is overwritten (P98h).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
TAG=sctx_d256_full169
LOG=lens/runs/SELF-CONTEXT-TIED.txt
{
  echo "================================================================================================"
  echo "SELF-CONTEXT / EXPOSURE-BIAS ARM (P98d-g)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt (written before this file existed)"
  echo "pre-flight (P98a-c): lens/runs/SELF-CONTEXT-PREFLIGHT.txt"
  echo "recipe = the twin ct_d256_full169_fix3 PLUS --self-context-frac 0.5 --self-context-rounds 4"
  echo "twin to beat: 35.26% by position (ep3), step-8 free-running 3.9%, pad-free talk 2.1% <unk>, 0/6 answers"
  echo "================================================================================================"
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
    --seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode tied \
    --max-windows 16900000 --method standard --out "$TAG" --ckpt-seconds 1800 \
    --self-context-frac 0.5 --self-context-rounds 4
  echo "ARM EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- chained decider 1/3: PAD-FREE TALK INSTRUMENT on the arm's best AND the twin, ONE run (P98f) ---"
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SELFCTX.txt" \
  CKPT_LIST="SELF-CONTEXT arm (frac 0.5, rounds 4; held-out by position as logged above)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the matched control, stored, held-out 35.26% by position)=lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- chained decider 2/3: FREE-RUNNING BY STEP on the SELF-CONTEXT arm's best (P98e) ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-SELFCTX.txt" \
  TALK_FAIR_LOGGED="0.0" \
  TALK_FAIR_EP="best (see the epoch lines above)" \
  TALK_FAIR_PREDICTIONS="lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt (P98e)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-ARM EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- chained decider 3/3: THE SAME INSTRUMENT ON THE TWIN = the control for that number ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-SELFCTX-TWIN.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 FINAL BEST (legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt (P98e control)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-TWIN EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
  echo "NEXT: grade P98d from lens/runs/chat/${TAG}_history.jsonl against the twin's three epoch rows;"
  echo "      P98e from the step-8 free-running line in TALK-FAIR-SELFCTX.txt against the twin's 3.9%"
  echo "      [TALK-FAIR-finalbest-d256-repro.txt] with TALK-FAIR-SELFCTX-TWIN.txt as the instrument control;"
  echo "      P98f BY EYE on the six prompt/true-reply/greedy blocks of PADFREE-TALK-SELFCTX.txt."
} >> "$LOG" 2>&1
