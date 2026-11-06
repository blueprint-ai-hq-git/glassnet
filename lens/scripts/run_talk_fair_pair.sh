#!/bin/bash
# THE MASK-FIX PAIR, ONE INSTRUMENT, ONE EPOCH (2026-10-22 18:5x self-wake).
# Predictions P70a-e written BEFORE this ran: lens/runs/MASKFIX-PAIR-PREDICTIONS-P70.txt
#   A = ct_d256_full169_3_ep1.npz      legacy by-token-id mask, ep1 (run ct_d256_full169_3, 17:55)
#   B = ct_d256_full169_fix3_best.npz  fixed by-position mask, ep1 (run ct_d256_full169_fix3, 18:50)
# Same corpus, windows, seed, step rule and architecture; the mask is the only difference.
# It writes its own outputs and its own .json files; nothing stored is overwritten. CPU only, nice 19.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export TALK_FAIR_PREDICTIONS="lens/runs/MASKFIX-PAIR-PREDICTIONS-P70.txt"
{
  echo "=== MASK-FIX PAIR (P70a-e) $(date '+%F %H:%M') ==="
  echo "--- A: legacy by-token-id mask, ep1 (logged legacy column 34.55%, lens/runs/TIED-FULLCORPUS-169M.txt line 61) ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_3_ep1.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-legacy169-ep1.txt" \
  TALK_FAIR_LOGGED=0.3455 TALK_FAIR_EP="ep1" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "exitA=$?"
  echo "--- B: fixed by-position mask, ep1 (logged legacy column 22.33%, lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt) ---"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-maskfix-ep1.txt" \
  TALK_FAIR_LOGGED=0.2233 TALK_FAIR_EP="ep1" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "exitB=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TALK-FAIR-MASKFIX-PAIR.txt 2>&1
