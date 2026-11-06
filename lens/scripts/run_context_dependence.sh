#!/bin/bash
# CONTEXT-DEPENDENCE runner (2026-10-22 20:3x self-wake). CPU ONLY.
# Predictions P75a-f were written BEFORE this file ran: lens/runs/CONTEXT-DEPENDENCE-PREDICTIONS-P75.txt
# Writes ONLY lens/runs/CONTEXT-DEPENDENCE.txt and ...-RUN.txt (appended). Nothing else is touched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CD_OUT="$PWD/lens/runs/CONTEXT-DEPENDENCE.txt"
export CKPT_LIST="SHIP CANDIDATE d256 maskfix full-corpus FINAL best (ep3, held-out 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz;d512 full-corpus ep1 best (held-out 33.17% by position)=lens/runs/chat/ct_d512_full169_fix3_best.npz"
{
  echo "=== CONTEXT-DEPENDENCE runner $(date '+%F %H:%M:%S') ==="
  echo "checkpoints: $CKPT_LIST"
  echo "  predictions: lens/runs/CONTEXT-DEPENDENCE-PREDICTIONS-P75.txt"
  for f in lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d512_full169_fix3_best.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/context_dependence_test.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/CONTEXT-DEPENDENCE-RUN.txt 2>&1
