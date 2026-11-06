#!/bin/bash
# Read the t11 baseline's per-epoch curve: the honest comparison for the ladder's 96-epoch runs
# must be the baseline's number at the SAME epoch count, not its 192-epoch final.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json
d = json.load(open("lens/runs/chat/t11_talk41_log.json"))
print("t11 log keys:", sorted(d.keys())[:14])
rows = d.get("epochs") or d.get("log") or d.get("history") or []
print("rows:", len(rows))
if rows:
    it = rows if isinstance(rows, list) else list(rows.items())
    for r in it[93:97] + it[-2:]:
        print(r)
PY