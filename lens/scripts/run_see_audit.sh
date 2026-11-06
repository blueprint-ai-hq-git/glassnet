#!/bin/bash
# Strict audit of the RUNTIME's see ship (the image -> class-word line, 23 extended concepts),
# on the same pinned datum. The see line and the concept/caption line are separate artifacts, so
# "image is 100%" must be claimed per-ship; the number this prints decides whether the see ship
# shares the 19/19 result or gets its own named dossier.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 VISION_AUDIT_N=60000
VISION_SHIP=ktest_real_s0_ep12.npz VISION_CKPT="$PWD/lens/runs/image/ktest_real_s0_ep12.npz" \
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/audit_strict_vision.py