#!/bin/bash
# THE EARLY TALK READ (P99b): the pad-free ship instrument on the SELF-CONTEXT arm's EPOCH-1 BEST, while epoch 2
# is still training on the GPU. CPU ONLY, taskset'd to cores 2,3 at nice 19 so it cannot disturb the trainer.
# Predictions: lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt (P99b), written BEFORE this file existed.
#
# WHY IT IS RUN NOW AND NOT LATER: the chained decider in the live job writes lens/runs/PADFREE-TALK-SELFCTX.txt
# on whatever the arm's best is AT THE END. This run reads the ep1-best AS IT IS NOW and writes its own NEW file
# (lens/runs/PADFREE-TALK-SELFCTX-EP1.txt). Nothing is overwritten; the twin is measured in the SAME run as the
# instrument's control, so the arm's column is only read if the twin's own block reproduces its stored values
# (mean <unk> 2.1%, 6/6 samples with a >=4-word run) [lens/runs/PADFREE-TALK-ATTN.txt twin block].
#
# NOT A SHIP DECISION: the ship decider is P98f on the arm's FINAL best. This is an early read on epoch 1.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
LOG=lens/runs/PADFREE-TALK-SELFCTX-EP1-RUN.txt
{
  echo "===================================================================================================="
  echo "EARLY TALK READ (P99b) — SELF-CONTEXT arm's EPOCH-1 BEST vs the stored TWIN control  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt (P99b), written before this runner existed"
  echo "note: the arm's epoch-1 best file is the one the trainer writes; if epoch 2 is later found better it will"
  echo "      be REPLACED by the trainer at the next epoch boundary — this read is of THIS state of that file."
  echo "===================================================================================================="
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SELFCTX-EP1.txt" \
  CKPT_LIST="SELF-CONTEXT arm EPOCH-1 BEST (frac 0.5 rounds 4 - held-out 25.74% by position at ep1, read while ep2 trains)=lens/runs/chat/sctx_d256_full169_best.npz;TWIN ct_d256_full169_fix3 (matched control, stored, held-out 35.26% by position)=lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "PADFREE-EP1 EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -5 "$LOG"
