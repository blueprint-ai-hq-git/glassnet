#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_vqa2.py 2>&1 | tail -14
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null; echo copied
echo; echo "=== big rung progress ==="
ls -la lens/runs/ladder/ladder_big.json 2>/dev/null || echo "  still running"
pgrep -af "ladder.py" | grep -v grep | head -2 || echo "  ladder finished"