#!/bin/bash
# speech_commands layout via the plain HTTP API (no huggingface_hub needed).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-data/bin/python3 -u - <<'PY'
import json, urllib.request
url = "https://huggingface.co/api/datasets/google/speech_commands/tree/main?recursive=true"
with urllib.request.urlopen(url, timeout=60) as r:
    items = json.load(r)
files = [i["path"] for i in items if i.get("type") == "file"]
print("  total files:", len(files))
for f in files[:10]:
    print("   ", f)
exts = {}
for f in files:
    e = f.rsplit(".", 1)[-1]
    exts[e] = exts.get(e, 0) + 1
print("  by extension:", exts)
PY