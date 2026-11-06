#!/bin/bash
# Train the readable spoken-word recognizer on the real Speech Commands features (CPU: the GPU is
# running the talk matrix). Honest gates: official val split + majority baseline printed, strict
# per-concept honesty audit, pruning cost reported.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u lens/scripts/train_speech.py > /tmp/train_speech.log 2>&1
echo "RC=$?"
tail -40 /tmp/train_speech.log