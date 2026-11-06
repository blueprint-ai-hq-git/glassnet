#!/bin/bash
# DECODE-SEARCH runner (2026-10-22 20:3x self-wake). CPU ONLY.
# Predictions P74a-f were written BEFORE this file ran: lens/runs/DECODE-SEARCH-PREDICTIONS-P74.txt
# It writes ONLY lens/runs/DECODE-SEARCH.txt and lens/runs/DECODE-SEARCH-RUN.txt (appended, never overwritten).
# No checkpoint, no stored run, no other instrument is touched; the GPU belongs to the d=512 arm in flight.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export SEARCH_OUT="$PWD/lens/runs/DECODE-SEARCH.txt"
export CKPT_LIST="SHIP CANDIDATE d256 maskfix full-corpus FINAL best (ep3, held-out 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz;d512 full-corpus ep1 best (held-out 33.17% by position)=lens/runs/chat/ct_d512_full169_fix3_best.npz"
{
  echo "=== DECODE-SEARCH runner $(date '+%F %H:%M:%S') ==="
  echo "NOTE: the FIRST launch of this runner (20:30:39) died at 1s: 'ValueError: windows must be 32 tokens wide,"
  echo "      got 33' — the beam search grew its own context past W before the forward. Fixed (the forward now sees"
  echo "      seq[-W:]), and the instrument now also writes its log INCREMENTALLY so a crash cannot lose the numbers"
  echo "      it already measured. No number was produced by the crashed attempt and nothing was overwritten."
  echo "checkpoints: $CKPT_LIST"
  echo "  predictions: lens/runs/DECODE-SEARCH-PREDICTIONS-P74.txt"
  for f in lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d512_full169_fix3_best.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_search_test.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/DECODE-SEARCH-RUN.txt 2>&1
