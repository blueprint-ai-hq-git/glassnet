#!/bin/bash
# run_gate_arm.sh -- PLAN item (d2c): the memorisation rule on the REAL recipe and the REAL corpus, as ONE
# queue entry. Predictions P59a-e are in lens/runs/TRAIN-GATES-PREDICTIONS.txt (APPENDIX), written BEFORE
# this line was queued.
#
# WHY IT IS QUEUED AND NOT RUN INLINE: it is a 12-epoch GPU arm (~40 minutes ungated, longer gated because
# every step pays one extra forward pass over the probe and one over the batch). The watchdog starts it only
# when no other train_chat is running, and the checkpoint per epoch means a shutdown cannot lose the run.
#
# THE ARM IS THE STORED sgd04 SEED-2 ARM WITH EXACTLY ONE CHANGE: --gates gap. Same corpus
# (data/chat/prepared_mix_ctrl_honest, val measured disjoint from train at the training window: 0 of 87,035
# windows overlap, lens/runs/VAL-TRAIN-LEAK-mix.txt), same d=256, same 1.5M windows, same seed 2, same mask,
# same step rule (--opt sgd --lr 0.4 --lr-decay 0.5). Its ungated twin is lens/runs/chat/opt_sgd04_s2_*,
# final honest masked val 0.284636858931064 (n=44,418).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
OUTF=lens/runs/GATE-ARM.txt
TAG="gate_sgd04_s2"
{
  echo "=== GATE ARM (real recipe) start $(date '+%F %H:%M:%S')"
  echo "    the stored opt_sgd04_s2 arm plus --gates gap (GapGate ema_band beta 0.5 tol 0.0),"
  echo "    probe 512 windows drawn once from the held-out split with seed 2, per-step log"
  echo "    lens/runs/chat/${TAG}_gates.jsonl, summary lens/runs/chat/${TAG}_gates_summary.json"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 12 --batch 1024 --lam-concept 0.3 \
    --seed 2 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --opt sgd --lr 0.4 --lr-decay 0.5 \
    --gates gap --probe-windows 512 --gate-log-every 200 --out "$TAG"
  echo "ARM EXIT=$?"
  echo "=== GATE ARM done $(date '+%F %H:%M:%S')"
  echo "    per-epoch rows: lens/runs/chat/${TAG}_history.jsonl   checkpoints: lens/runs/chat/${TAG}_ep*.npz"
} >> "$OUTF" 2>&1
