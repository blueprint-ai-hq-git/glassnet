#!/bin/bash
# run_camera_fashion.sh — regenerate the camera clip with per-frame questions, then verify it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/virtual_camera.py \
  --model lens/runs/vl/cam_base_ep6.npz --source fashion --frames 60 --fps 8 \
  --device cpu --out cam_fashion.mp4 2>&1 | tail -10

cp lens/runs/video/cam_fashion.mp4 /home/yahya/cam_fashion.mp4
FF=/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg
$FF -y -loglevel error -i lens/runs/video/cam_fashion.mp4 -vf "select=eq(n\,13)" -vframes 1 \
  /home/yahya/cam_frame13.png
ls -la /home/yahya/cam_fashion.mp4 /home/yahya/cam_frame13.png
