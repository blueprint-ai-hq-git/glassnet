#!/bin/bash
# DRIFT DISCRIMINATOR — RUN A and RUN B of ONE command, then the text-only grader.
# Predictions P107a-g were written BEFORE this file existed: lens/runs/DRIFT-DISCRIMINATOR-PREDICTIONS-P107.txt
# Log: lens/runs/DRIFT-DISCRIMINATOR.txt (appended to, never overwritten; the grader's output is its last block
# and is also kept standalone in lens/runs/DRIFT-NUMBERS.txt).
#
# WHY: the 6-epoch arm's epoch 1 did not reproduce the stored twin's epoch 1 (|d| = 0.0687 points of held-out
# accuracy, 8.68e-05 of train loss, unlike a val-pass artefact because the TRAIN numbers differ too) — see
# lens/runs/GRADE-P102.txt section 6. Two causes, both named and neither claimed: (i) a code-path/RNG change
# since the twin ran on 2026-10-22 18:27, (ii) non-bit-reproducible GPU float32 reductions. THIS RUNS THE SAME
# COMMAND TWICE and compares the two rows field by field: a difference says (ii); a bit-identical pair says the
# command reproduces itself on this machine and puts (i) on the table instead.
#
# THE ONLY DIFFERENCE BETWEEN THE TWO RUNS IS --out: identical corpus, d, window, epochs, batch, lr, decay,
# concept weight, seed, device, mask flags, tied path and window cap, and identical ONLY because
# lr(ep1) does not depend on --epochs (train_chat.py line 1046 uses max(a.epochs-1,1)).
#
# ENV OVERRIDES exist for the CPU/tiny-corpus wiring pre-flight only (lens/scripts/drift_preflight.sh); a run
# that uses them prints them in its header, so a tiny pre-flight can never be mistaken for the real thing.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

DATA_DIR="${DRIFT_DATA_DIR:-$PWD/data/chat/prepared_mix_ctrl_honest}"
export CHAT_DATA_DIR="$DATA_DIR"
PREFIX="${DRIFT_PREFIX:-drift_rerun}"
DEVICE="${DRIFT_DEVICE:-gpu}"
D="${DRIFT_D:-256}"
WINDOWS="${DRIFT_WINDOWS:-16900000}"
EPOCHS="${DRIFT_EPOCHS:-1}"
CKPT="${DRIFT_CKPT_SECONDS:-1800}"
LOG="${DRIFT_LOG:-lens/runs/DRIFT-DISCRIMINATOR.txt}"
PY="${DRIFT_PYTHON:-$PWD/.venv-gpu/bin/python3}"

# the same guard the other runners carry: a ';' in a CKPT_LIST label is this project's most expensive format bug
# ([BUG-CKPTLIST-LABEL-SEMICOLON]). This runner builds no CKPT_LIST, but it refuses loudly if one is exported
# into it by accident, because that is how the bug reached a live job on 2026-10-25.
if [ -n "$CKPT_LIST" ]; then
  "$PY" -c 'import os,sys; items=os.environ["CKPT_LIST"].split(";"); bad=[i for i in items if "=" not in i]; print("CKPT_LIST guard: %d item(s)" % len(items)); print("REFUSING: item(s) with no = (a label may not contain the ; separator): %r" % bad) if bad else print("CKPT_LIST guard: PASS"); sys.exit(3 if bad else 0)'
  if [ $? -ne 0 ]; then
    echo "CKPT_LIST GUARD REFUSED - NOTHING WAS STARTED  $(date '+%F %H:%M:%S')"
    exit 3
  fi
fi

{
  echo "================================================================================================"
  echo "DRIFT DISCRIMINATOR — RUN A then RUN B, ONE command, only --out differs   $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/DRIFT-DISCRIMINATOR-PREDICTIONS-P107.txt (written before this file existed)"
  echo "question: is the twin-vs-6ep gap at ep1 (0.0687 points, GRADE-P102 section 6) RUN-TO-RUN NOISE (ii)"
  echo "          or a CODE/RNG PATH CHANGE (i)?  A-vs-B difference => (ii).  Bit-identical A and B while the"
  echo "          twin's row is not reproduced => (i)."
  echo "config: data=$DATA_DIR d=$D window=32 epochs=$EPOCHS batch=256 lr=0.4 decay=0.5 lam_concept=0.3 seed=0"
  echo "        device=$DEVICE assistant-only train-turn-end direct=tied max-windows=$WINDOWS ckpt-seconds=$CKPT"
  echo "        tags: ${PREFIX}_a / ${PREFIX}_b   (the ONLY argv difference is --out)"
  if [ "$DATA_DIR" != "$PWD/data/chat/prepared_mix_ctrl_honest" ] || [ "$D" != "256" ] || [ "$EPOCHS" != "1" ] \
     || [ "$DEVICE" != "gpu" ] || [ "$WINDOWS" != "16900000" ]; then
    echo "  *** OVERRIDDEN CONFIG — THIS IS A WIRING PRE-FLIGHT, NOT THE REAL DISCRIMINATOR:"
    echo "      DRIFT_DATA_DIR=$DRIFT_DATA_DIR DRIFT_D=$DRIFT_D DRIFT_EPOCHS=$DRIFT_EPOCHS"
    echo "      DRIFT_DEVICE=$DRIFT_DEVICE DRIFT_WINDOWS=$DRIFT_WINDOWS DRIFT_PREFIX=$DRIFT_PREFIX"
    echo "      NO P107 VERDICT MAY BE READ FROM A RUN THAT PRINTS THIS LINE."
  fi
  echo "================================================================================================"
  echo "--- RUN A  $(date '+%F %H:%M:%S') ---"
  "$PY" -u lens/scripts/train_chat.py \
    --d "$D" --window 32 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device "$DEVICE" --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows "$WINDOWS" --method standard --out "${PREFIX}_a" \
    --ckpt-seconds "$CKPT"
  echo "RUN A EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- RUN B (identical argv except --out)  $(date '+%F %H:%M:%S') ---"
  "$PY" -u lens/scripts/train_chat.py \
    --d "$D" --window 32 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device "$DEVICE" --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows "$WINDOWS" --method standard --out "${PREFIX}_b" \
    --ckpt-seconds "$CKPT"
  echo "RUN B EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- the grader (text only: no checkpoint is opened, no model, no GPU)  $(date '+%F %H:%M:%S') ---"
  DRIFT_TAG_A="${PREFIX}_a" DRIFT_TAG_B="${PREFIX}_b" "$PY" -u lens/scripts/drift_numbers.py
  echo "GRADER EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 8 "$LOG"
