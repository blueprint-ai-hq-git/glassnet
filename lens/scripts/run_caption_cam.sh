#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/caption_camera.py \
  --frames "${FR:-48}" --out "${OUT:-caption_cam.mp4}" --device cpu 2>&1 | tail -30
