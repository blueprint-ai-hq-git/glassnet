#!/bin/bash
# Extract the readable feature table from all 8-word clips (~31k) on CPU. Background job.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u lens/scripts/speech_data.py > /tmp/speech_data.log 2>&1
echo "RC=$?"
tail -16 /tmp/speech_data.log