#!/bin/bash
# SELF-CONTEXT ARM — DECIDER REPAIR (P101a-e). The arm is DONE (ARM EXIT=0, 14:35:05); this runner re-runs ONLY
# the two chained deciders that the arm's own job could not deliver, because its environment was wrong, not the
# arm. Full argument: lens/runs/SELFCTX-DECIDER-REPAIR-PREDICTIONS-P101.txt (written BEFORE this file existed).
#   * decider 1: the pad-free v3 ship instrument (THE DECIDER, P98f) — old CKPT_LIST label contained ';', the
#     instrument's own item separator, so it died with ValueError before it read anything.
#   * decider 2: talk_fair_test.py (P98e, the lever's own falsifier) — old run passed TALK_FAIR_LOGGED="0.0" for
#     an arm whose legacy column is 19.28% (its own epoch-3 row), so its reproduction control printed FAIL.
# CPU ONLY, taskset'd to cores 2,3 at nice 19. NO TRAINER, NO GPU, NO ARM RE-RUN. Nothing stored is written to:
# lens/runs/TALK-FAIR-SELFCTX.txt and TALK-FAIR-SELFCTX-TWIN.txt are left exactly as the arm wrote them.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
TAG=sctx_d256_full169
LOG=lens/runs/SELFCTX-DECIDER-REPAIR.txt
{
  echo "===================================================================================================="
  echo "SELF-CONTEXT DECIDER REPAIR (P101a-e)   $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SELFCTX-DECIDER-REPAIR-PREDICTIONS-P101.txt (written before this file existed)"
  echo "the arm is NOT re-run: ARM EXIT=0 at 14:35:05, three epoch rows, done."
  echo "===================================================================================================="
  echo "--- repair 1/2: PAD-FREE SHIP INSTRUMENT on the arm's FINAL BEST and the stored TWIN, ONE run (P98f) ---"
  echo "    label has NO ';' (that is the bug being repaired) and the twin is inside this same run as the control"
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-SELFCTX.txt" \
  CKPT_LIST="SELF-CONTEXT arm FINAL BEST (frac 0.5 rounds 4 - held-out 28.21% by position at ep3 - ep3 was its best)=lens/runs/chat/${TAG}_best.npz;TWIN ct_d256_full169_fix3 (the matched control - stored - held-out 35.26% by position)=lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "PADFREE EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "--- repair 2/2: FREE-RUNNING BY STEP on the arm's FINAL BEST (P98e) with the CORRECT logged legacy value ---"
  echo "    TALK_FAIR_LOGGED is a FRACTION: the arm's own ep3 val_masked_acc 0.19280471880769057 -> 0.1928"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-SELFCTX-RERUN.txt" \
  TALK_FAIR_LOGGED="0.1928" \
  TALK_FAIR_EP="ep3 FINAL BEST (the arm's own jsonl: legacy by-id 19.28%, by-position 28.21%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/SELFCTX-DECIDER-REPAIR-PREDICTIONS-P101.txt (P101c/P101d)" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "TALKFAIR-RERUN EXIT=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
echo "--- runner log tail ---"
tail -n 12 "$LOG"
