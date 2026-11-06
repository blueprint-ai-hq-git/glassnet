#!/bin/bash
# run_vl_smoke.sh -- verify the whole VL path on a small slice before spending GPU time.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
VL_DATA="${VL_DATA:-fashion}" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --data "${VL_DATA:-fashion}" --d 64 --epochs 2 --batch 256 --device cpu \
  --max-train 6000 --out vlsmoke 2>&1 | tail -45