#!/bin/bash
# run_gate_test.sh -- the measurement half of PLAN item (d2b): does the memorisation rule FIRE inside the
# SHIPPED trainer, on a corpus where memorisation is possible? Predictions first:
# lens/runs/TRAIN-GATES-PREDICTIONS.txt (P58b, P58d, P58e, P58f, P58h, P58i, P58j).
#
# ARMS (all CPU, d=128, window 32, batch 256, lr 0.4, nice 19, 2 BLAS threads - the GPU is busy with the
# 16-arm comparison and must not be disturbed):
#   M-OFF      prepared_gate_small, no gate                     -> the baseline the gate is judged against
#   M-ON       prepared_gate_small, --gates gap                 -> must FIRE (P58b) and not damage (P58d)
#   M-ON2      identical to M-ON, same seed, different --out    -> the consistency check (P58f)
#   L-ON       prepared_clean, --max-windows 20000, --gates gap -> the cost on a corpus that CAN be learned
#   M-RELGRAD  prepared_gate_small, --gates gap --opt relgrad   -> the rule and OUR step rule together (P58j)
#
# 80 epochs of the small corpus is ~376 steps: enough passes over 1,197 windows for the train loss to fall
# well below the held-out loss, which is the memorisation the gate is supposed to see.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/GATE-TEST.txt
SMALL="$PWD/data/chat/prepared_gate_small"
CLEAN="$PWD/data/chat/prepared_clean"
COMMON="--d 128 --window 32 --batch 256 --lr 0.4 --lam-concept 0.3 --seed 0 --device cpu"

{
  echo "=== GATE TEST START $(date '+%F %H:%M:%S')"
  echo "--- M-OFF: memorisable corpus, NO gate (baseline)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
    --epochs 80 --out gate_m_off
  echo "M-OFF EXIT=$?"
  echo "--- M-ON: memorisable corpus, --gates gap (THE ITEM: it must fire here)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
    --epochs 80 --gates gap --probe-windows 256 --out gate_m_on
  echo "M-ON EXIT=$?"
  echo "--- M-ON2: identical to M-ON, same seed (the consistency check, P58f)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
    --epochs 80 --gates gap --probe-windows 256 --out gate_m_on2
  echo "M-ON2 EXIT=$?"
  echo "--- L-ON: prepared_clean, 20,000 windows, --gates gap (the cost on a learnable corpus)"
  CHAT_DATA_DIR="$CLEAN" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
    --epochs 3 --max-windows 20000 --gates gap --probe-windows 256 --out gate_l_on
  echo "L-ON EXIT=$?"
  echo "--- M-RELGRAD: memorisable corpus, --gates gap --opt relgrad (the rule with OUR step rule, P58j)"
  CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
    --epochs 20 --gates gap --probe-windows 256 --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02 \
    --out gate_m_relgrad
  echo "M-RELGRAD EXIT=$?"
  echo "=== GATE TEST DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
