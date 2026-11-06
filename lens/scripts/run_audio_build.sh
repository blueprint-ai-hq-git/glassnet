#!/bin/bash
# AUDIO line: build the decoded+STFT dataset from Yahya's own oggs (CPU, nice/ionice low prio).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/audio_data.py 2>&1 \
  | tee lens/runs/audio_build.log