#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/show_big1.py 2>&1 | tail -55