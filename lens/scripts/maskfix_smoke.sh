#!/bin/bash
# maskfix_smoke.sh — VERIFICATION for BUGS.md [BUG-MASK-TOKEN-ID] (self-wake 2026-10-22 18:1x).
# Predictions: lens/runs/MASKFIX-PREDICTIONS.txt (P66a-h), written BEFORE the edit.
# Pre-change copy: lens/runs/_tmp/train_chat_prefix_maskfix.py
#
# WHAT IT CHECKS (P66h, the check that must FAIL if the fix is wrong):
#   the fixed trainer, on data/chat/prepared_mix_ctrl_honest, must print
#     ASST(by-position) ... (n=58191)   in every epoch line   <- computed independently in
#                                                               lens/runs/MASK-INDEX-BUG.txt
#     MASKED(legacy by-id) ... (n=44418)                      <- the count every earlier arm reported
#   and on the TRAINING side "BY POSITION" must read ~66.9% where LEGACY reads ~51.0%.
# It is CPU-only (d=64, 4,000 windows, 1 epoch) so it cannot touch the GPU arm ct_d256_full169_3.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
{
  echo "=== MASKFIX SMOKE $(date '+%F %H:%M:%S') ==="
  echo "expected (independently computed, lens/runs/MASK-INDEX-BUG.txt):"
  echo "   val  ASST(by-position) n=58191      MASKED(legacy by-id) n=44418"
  echo "   train BY POSITION ~66.9% of built windows, LEGACY ~51.0%"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 64 --window 32 --epochs 1 --batch 64 --lr 0.4 --lam-concept 0.3 \
    --seed 0 --device cpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 4000 --out maskfix_smoke
  echo "SMOKE EXIT=$?"
  echo "=== MASKFIX SMOKE done $(date '+%F %H:%M:%S') ==="
} > lens/runs/MASKFIX-SMOKE.txt 2>&1
