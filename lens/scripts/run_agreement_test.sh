#!/bin/bash
# run_agreement_test.sh -- the measurement half of PLAN item (d2d): does the AGREEMENT number (LEARN-NOT-
# MEMORISE.md 2b) separate a corpus that CAN be memorised from one that CAN be learned, inside the SHIPPED
# trainer? Predictions first: lens/runs/AGREEMENT-GATE-PREDICTIONS.txt (P60c-P60k).
#
# THE ARMS ARE THE GAP GATE'S OWN ARMS, so the two rules are compared on the same data, the same model and the
# same step count (the gap gate refused 74/400 = 0.185 on M and 2/237 = 0.008 on L, lens/runs/GATE-REPORT.txt):
#   M = data/chat/prepared_gate_small  --epochs 80            -> 400 steps  (CAN be memorised)
#   L = data/chat/prepared_clean --max-windows 20000 --epochs 3 -> 237 steps (CAN be learned)
# All CPU, nice 19, 2 BLAS threads: the GPU is busy with the 16-arm comparison and must not be disturbed.
#
# SCREEN (seed 0, all three detectors on both corpora, plus a determinism repeat of the default):
#   agree_m_vs_s0 / agree_m_vs_s0b   default 'vs_probe'   M  (the repeat is P60c)
#   agree_l_vs_s0                    default 'vs_probe'   L
#   agree_m_ema_s0 / agree_l_ema_s0  'ema_band' tol 0.0   M / L
#   agree_m_lev_s0 / agree_l_lev_s0  'level'    tol 0.5   M / L
# The seeds 1 and 2 arms for whichever detector separates are appended below ONCE the screen has been read -
# three seeds before anything is called a win.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/AGREEMENT-TEST.txt
SMALL="$PWD/data/chat/prepared_gate_small"
CLEAN="$PWD/data/chat/prepared_clean"
COMMON="--d 128 --window 32 --batch 256 --lr 0.4 --lam-concept 0.3 --device cpu"
M="--epochs 80 --seed 0"
L="--epochs 3 --max-windows 20000 --seed 0"

{
  echo "=== AGREEMENT TEST START $(date '+%F %H:%M:%S')"
  echo "--- agree_m_vs_s0: M, DEFAULT detector vs_probe (the item's own mechanism)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $M \
    --gates agreement --out agree_m_vs_s0
  echo "agree_m_vs_s0 EXIT=$?"
  echo "--- agree_m_vs_s0b: identical to agree_m_vs_s0, same seed (P60c determinism)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $M \
    --gates agreement --out agree_m_vs_s0b
  echo "agree_m_vs_s0b EXIT=$?"
  echo "--- agree_l_vs_s0: L, DEFAULT detector vs_probe"
  CHAT_DATA_DIR="$CLEAN" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $L \
    --gates agreement --out agree_l_vs_s0
  echo "agree_l_vs_s0 EXIT=$?"
  echo "--- agree_m_ema_s0: M, ema_band tol 0.0 (the gap gate's detector-1 analogue)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $M \
    --gates agreement --agree-variant ema_band --out agree_m_ema_s0
  echo "agree_m_ema_s0 EXIT=$?"
  echo "--- agree_l_ema_s0: L, ema_band tol 0.0"
  CHAT_DATA_DIR="$CLEAN" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $L \
    --gates agreement --agree-variant ema_band --out agree_l_ema_s0
  echo "agree_l_ema_s0 EXIT=$?"
  echo "--- agree_m_lev_s0: M, level tol 0.5 (a FIXED threshold, measured not believed)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $M \
    --gates agreement --agree-variant level --agree-tol 0.5 --out agree_m_lev_s0
  echo "agree_m_lev_s0 EXIT=$?"
  echo "--- agree_l_lev_s0: L, level tol 0.5"
  CHAT_DATA_DIR="$CLEAN" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON $L \
    --gates agreement --agree-variant level --agree-tol 0.5 --out agree_l_lev_s0
  echo "agree_l_lev_s0 EXIT=$?"
  echo "=== AGREEMENT TEST SCREEN DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
