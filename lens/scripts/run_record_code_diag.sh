#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_code_diag.py 2>&1 | tail -16
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null; echo copied
echo; echo "=== background jobs ==="
pgrep -af "ladder.py|run_medical_seeds|train_vl" | grep -v grep | cut -c1-95 || echo "  none"
ls -la --time-style=+%H:%M lens/runs/ladder/ladder_big.json lens/runs/vl/med_cc_s*.json 2>/dev/null || true