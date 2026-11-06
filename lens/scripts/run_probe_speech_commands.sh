#!/bin/bash
# speech_commands: find the REAL file layout in the repo and download accordingly (the include
# pattern "data/*" matched nothing -> empty dir, verified).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
HF=/home/yahya/.hermes/installs/15d969a29be1af88/environments/7c7750f1e3d5471e920a055130314e9b/venv/bin/hf
.venv-data/bin/python3 -u - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("google/speech_commands", repo_type="dataset")
print("  files in google/speech_commands (first 12):")
for f in files[:12]:
    print("   ", f)
print("  total files:", len(files))
exts = {}
for f in files:
    exts[f.split(".")[-1]] = exts.get(f.split(".")[-1], 0) + 1
print("  by extension:", exts)
PY