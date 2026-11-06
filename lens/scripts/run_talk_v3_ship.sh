#!/bin/bash
# SHIP-TALK v3 runner (2026-10-22 19:5x). THE TALK TEST ON THE ARTIFACT THAT WOULD SHIP: the FINAL best
# checkpoint of the mask-fix full-corpus d=256 arm (ct_d256_full169_fix3_best.npz = ep3, held-out 35.26% by
# position, lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt line 69), against the same arm's ep1 checkpoint that v2
# already reported on, on REAL UNPADDED 32-token prompts (the v2 padding fix, lens/runs/PADFREE-TALK.txt).
#
# PREDICTIONS P72a-h were written BEFORE this file existed: lens/runs/SHIP-TALK-PREDICTIONS-P72.txt.
# IT WRITES ONLY: lens/runs/SHIP-TALK-v3.txt, lens/runs/SHIP-TALK-v3-RUN.txt.
# It does NOT touch lens/runs/PADFREE-TALK.txt (the v2 record) or any checkpoint.
# It never edits the model, never trains, never uses the GPU: CPU only, 2 BLAS threads, nice 19, and it is
# launched inside a memory-capped scope so it cannot push the d=512 arm in flight over this machine's ceiling.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PADFREE_OUT="$PWD/lens/runs/SHIP-TALK-v3.txt"
export CKPT_LIST="SHIP CANDIDATE maskfix full-corpus d256 FINAL best (ep3, held-out 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz;maskfix full-corpus d256 ep1 (held-out 32.52%, the epoch v2 reported on)=lens/runs/chat/ct_d256_full169_fix3_ep1.npz"
{
  echo "=== SHIP-TALK v3 runner $(date '+%F %H:%M:%S') ==="
  echo "NOTE: the FIRST launch of this runner (19:45) died in 11s with FileNotFoundError because its labels"
  echo "      contained '=' and the parser split on the first '='; labels no longer contain '=' and the parser"
  echo "      now rsplit()s. No number was produced by that attempt; nothing was overwritten."
  echo "checkpoints: $CKPT_LIST"
  for f in lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d256_full169_fix3_ep1.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/SHIP-TALK-v3-RUN.txt 2>&1
