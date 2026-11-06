#!/bin/bash
# run_chat_big.sh — the talking model needs more training and more capacity.
#
# Run 1: d=128, 1.65M params, 3 epochs -> val ppl 353, acc 11.08%, concepts 5/8.
#   It learned the FORMAT (Human: -> Assistant:) but the replies are not coherent.
#   That is under-trained and under-sized, not broken.
#
# The ceiling measurement says a 10M model is ~3.5 hours to Chinchilla-optimal on this GPU, and the
# corpus holds 13.3M tokens. So this run goes bigger AND longer:
#   d=384 with 9 concepts ~ 4.9M params, and 12 epochs.
#
# Pre-registered predictions (written before running):
#   1. val perplexity falls well below 100 (from 353). If it does not, the problem is not training
#      time and I will say so.
#   2. Generated replies become locally coherent (real words in real short phrases), but the model
#      will still contradict itself over longer spans. A 5M model does not hold a topic.
#   3. The concept pass rate stays at or above 5/8; the three firing-floor failures stay failures
#      unless their firing rate genuinely rises, and the floor does NOT move.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "run_distill.py|train_chat.py" > /dev/null; do
  sleep 30
done
echo "GPU free; starting the bigger chat run"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 384 --window 8 --epochs "${CB_EPOCHS:-12}" --batch 256 --lr 0.4 --lam-concept 0.3 \
  --device gpu --out big1 --max-windows 1500000 2>&1 | tail -90
