#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk.py \
  --d "${D:-32}" --epochs "${EP:-6}" --n "${NIMG:-4000}" --device cpu \
  --out "${OUT:-seetalk_tiny}" 2>&1 | tail -32
