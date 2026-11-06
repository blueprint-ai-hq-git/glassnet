#!/bin/bash
# HH-RLHF license check (the base talk corpus source) + download progress.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-data/bin/python3 -u - <<'PY'
import json, urllib.request
for rid in ("Anthropic/hh-rlhf", "Anthropic/hh-rlhf", "openai/gsm8k"):
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{rid}", timeout=30) as r:
            d = json.load(r)
        print(f"  {rid:<28} license: {(d.get('cardData') or {}).get('license', 'NOT DECLARED')}")
    except Exception as e:
        print(f"  {rid:<28} ERROR {type(e).__name__}")
PY
echo "=== download progress:"
tail -5 /tmp/download_external.log
du -sh data/external/* 2>/dev/null