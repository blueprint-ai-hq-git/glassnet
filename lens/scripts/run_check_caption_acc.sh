#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 - <<'PY'
import json
d = json.load(open("lens/runs/video/caption_cam_log.json"))
proper = sum(1 for l in d["lines"] if l["noun"] == l["truth"])
naive = sum(1 for l in d["lines"] if l["ok"])
print(f"  proper full-name match: {proper}/48 = {proper/48*100:.1f}%  (video overlay used "
      f"first-word match: {naive}/48)")
mult = [l for l in d["lines"] if l["truth"] in ("ankle boot") or " " in l["truth"]]
print(f"  multi-word truths present: {len(mult)}  -> first-word rule penalises 'ankle boot' "
      f"predictions")
PY
