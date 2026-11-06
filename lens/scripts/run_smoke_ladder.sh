#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u lens/scripts/smoke_ladder_timeout.py 2>&1 | tail -25
