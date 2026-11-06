#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_ktest.py 2>&1 | tail -3
.venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -6
import json
d = json.load(open("lens/runs/image/lad_image_cls_log.json"))
print("ladder image log keys:", list(d.keys())[:8])
PY
cp LOG.md /home/yahya/ 2>/dev/null; cp API.md /home/yahya/ 2>/dev/null; echo synced; date '+%H:%M'
