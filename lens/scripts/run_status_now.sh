#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
date '+NOW: %F %H:%M %Z (%A)'
echo
echo "=== RUNNING ==="
pgrep -af "ladder.py|train_code|train_vl|train_chat" | grep -v grep | cut -c1-90 || echo "  nothing"
echo
echo "=== THE VIDEO FILES ==="
ls -la --time-style=+%m-%d_%H:%M lens/runs/video/ 2>/dev/null
echo
echo "=== LADDER ==="
for r in tiny small medium big; do
  f="lens/runs/ladder/ladder_$r.json"
  [ -f "$f" ] && .venv-gpu/bin/python3 -c "
import json;t=json.load(open('$f'));n=sum(x['passed'] for x in t)
print(f'  $r: {n}/{len(t)}')"
done
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
