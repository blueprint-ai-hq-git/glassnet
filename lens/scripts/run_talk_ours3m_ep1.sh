#!/bin/bash
# PAD-FREE TALK TEST on OUR METHOD's first completed-epoch artifact (P86f/P86g).
# Instrument: lens/scripts/talk_fair_v3_ship.py, ENV-DRIVEN checkpoint list (no file edits), CPU ONLY, no GPU,
# no training, 2 BLAS threads, nice 19, pinned to cores 4,5 so the attention arm on cores 0,1 is not disturbed.
# IT WRITES ONLY lens/runs/PADFREE-TALK-OURS3M.txt and its runner log. It does NOT touch PADFREE-TALK.txt,
# SHIP-TALK-v3.txt, TALK-FAIR-finalbest-d256-repro.txt or any checkpoint.
# Predictions: lens/runs/OURS3M-GRADE-PREDICTIONS-P86.txt (P86f, P86g) — written before this file ran.
# The comparator is loaded in the SAME run by the SAME instrument so the two <unk> shares are measured once,
# not quoted across two different runs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-OURS3M.txt"
# NOTE (bug fixed before the first real run): the CKPT_LIST separator is ';' and this file's first version put
# semicolons INSIDE the labels, so the parser split mid-label and died with ValueError in 1s (no number
# produced, nothing overwritten). Labels now contain no ';'.
export CKPT_LIST="OUR METHOD 3M ep1 best (relgrad + gap gate + agreement gate, held-out by-position 12.68%)=lens/runs/chat/ours_d256_3M_2ep_best.npz;TWIN maskfix full-corpus best (plain SGD, no gates, held-out by-position 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz"
{
  echo "=== PAD-FREE TALK OURS-3M runner $(date '+%F %H:%M:%S') ==="
  echo "checkpoints: $CKPT_LIST"
  for f in lens/runs/chat/ours_d256_3M_2ep_best.npz lens/runs/chat/ct_d256_full169_fix3_best.npz; do
    stat -c '  %y  %s bytes  %n' "$f"
  done
  start=$(date +%s)
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/PADFREE-TALK-OURS3M-RUN.txt 2>&1
