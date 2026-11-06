#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_medical.py 2>&1 | tail -14
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null; echo copied
echo; echo "=== big ladder rung ==="
ls -la lens/runs/ladder/ladder_big.json 2>/dev/null || echo "  still running"
pgrep -af "ladder.py" | head -1 | cut -c1-70 || echo "  finished"