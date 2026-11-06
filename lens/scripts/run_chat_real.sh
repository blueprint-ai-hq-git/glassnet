#!/bin/bash
# run_chat_real.sh — the real conversational training run, queued behind the GPU job.
#
# Sized from the DATA: 13.3M tokens. d=128 with 9 concepts is ~1.1M parameters, which is
# ~12 tokens/parameter -- well trained for this corpus. The smoke run proved the pipeline works;
# this is the run whose output actually matters.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU to be free..."
while pgrep -f "run_distill.py|run_blocker.py|run_scaling.sh" > /dev/null; do
  sleep 30
done
echo "GPU free; starting the real chat run"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 8 --epochs 3 --batch 256 --lr 0.4 --lam-concept 0.3 \
  --device gpu --out real1 --max-windows 900000 2>&1 | tail -80
