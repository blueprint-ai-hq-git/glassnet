#!/bin/bash
# run_image_smoke.sh -- verify the image path on a small slice before spending GPU time.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
  --d 64 --epochs 2 --batch 256 --train-n 6000 --device cpu --out imgsmoke 2>&1 | tail -50