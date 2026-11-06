#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_image_seeds.py 2>&1 | tail -20
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null; echo "docs copied to /home/yahya/"