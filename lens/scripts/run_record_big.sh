#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u lens/scripts/record_big_fixed.py 2>&1 | tail -22
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null && echo copied
