#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "=== SMOKE: tiny run of --extra real (train_n 4000, 2 epochs, cpu) ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
  --train-n 4000 --epochs 2 --device cpu --extra real --extra-k 10 --no-plain \
  --out ktest_smoke 2>&1 | grep -E "EXTRA|READABLE|ep 2|TEST|saved|Error|error|Traceback" | head -8
