#!/bin/bash
# IMAGE line provenance, done properly: recursive search for test_acc in every image/seetalk log
# (they nest under results.*), plus a repo-wide hunt for the "84.36%" figure I quoted earlier --
# if it has no source file, it does not get quoted again.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json, glob
best = None
def walk(o, f, path=""):
    global best
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "test_acc" and isinstance(v, (int, float)):
                if best is None or v > best[1]:
                    best = (f, v, path)
            walk(v, f, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[-3:]):
            walk(v, f, f"{path}[-{3-i}]")
for f in glob.glob("lens/runs/image/*.json") + glob.glob("lens/runs/seetalk/*.json") + \
         glob.glob("lens/runs/vl/*.json"):
    try:
        walk(json.load(open(f)), f)
    except Exception:
        continue
print("IMAGE/medical best test_acc across logs:", best)
PY
echo "=== hunt for the 84.36 figure:"
grep -rl "84.36\|0.8436" /run/media/yahya/HARD DISK 4/GlassNet/lens/ 2>/dev/null | head -5