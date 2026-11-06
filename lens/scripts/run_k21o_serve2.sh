#!/bin/bash
# Serve-check the new 19/19 image ship through the REAL runtime: one_chat reads paths from stdin
# and routes by extension; the newest seetalk ship (k21o_mirror) is picked by build_default().
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
printf 'hello , how are you ?\n/home/yahya/cam_frame13.png\n/home/yahya/amar_slang_p1.ogg\n/home/yahya/cam_frame12.png\n' |
  .venv-gpu/bin/python3 -u lens/scripts/one_chat.py