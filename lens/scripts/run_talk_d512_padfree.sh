#!/bin/bash
# run_talk_d512_padfree.sh (2026-10-22 21:2x self-wake). THE DECIDER ON THE d=512 CAPACITY ARM.
# PREDICTIONS P76a-f: lens/runs/D512-PADFREE-PREDICTIONS-P76.txt — written BEFORE this file existed.
# Same instrument as the ship decision (lens/scripts/talk_fair_v3_ship.py, unedited, env-driven), three
# checkpoints: d512 fix3 FINAL best (ep3) / d256 fix3 FINAL best (ep3) / d256 fix3 ep1 (reproduction control).
# WRITES ONLY: lens/runs/D512-PADFREE-TALK.txt and lens/runs/D512-PADFREE-TALK-RUN.txt.
# Does NOT touch lens/runs/PADFREE-TALK.txt, SHIP-TALK-v3.txt (=the stored control), any checkpoint, or any
# training file. CPU only, 2 BLAS threads, nice 19, no GPU use.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PADFREE_OUT="$PWD/lens/runs/D512-PADFREE-TALK.txt"
export CKPT_LIST="D512 FULL169 fix3 FINAL best ep3 (by-position 37.18%, asst ppl 29.35, 15.5M params)=lens/runs/chat/ct_d512_full169_fix3_best.npz;D256 FULL169 fix3 FINAL best ep3 (by-position 35.26%, asst ppl 32.77, 5.9M params)=lens/runs/chat/ct_d256_full169_fix3_best.npz;REPRODUCTION CONTROL d256 fix3 ep1 (stored in PADFREE-TALK.txt and SHIP-TALK-v3.txt)=lens/runs/chat/ct_d256_full169_fix3_ep1.npz"
{
  echo "=== D512-PADFREE talk runner $(date '+%F %H:%M:%S') ==="
  echo "predictions: lens/runs/D512-PADFREE-PREDICTIONS-P76.txt (P76a-f)"
  echo "checkpoints: $CKPT_LIST"
  for f in lens/runs/chat/ct_d512_full169_fix3_best.npz lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d256_full169_fix3_ep1.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  echo "mechanical checks: d512_best == d512_ep3 $(md5sum lens/runs/chat/ct_d512_full169_fix3_best.npz lens/runs/chat/ct_d512_full169_fix3_ep3.npz | awk '{print $1}' | uniq | wc -l) unique hash(es) (1 = identical)"
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/D512-PADFREE-TALK-RUN.txt 2>&1
