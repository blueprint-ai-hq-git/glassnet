#!/bin/bash
# BENCH ACCURACY PROVENANCE — find the recorded accuracy number for each line's SHIPPED ship and
# the datum it was measured on, straight from the logs/audits (never from memory).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== CODE: t8f_code24_ep32 log (val acc):"
.venv-gpu/bin/python3 -u - <<'PY'
import json, os, glob
for pat in ("lens/runs/code/t8f_code24*log*.json", "lens/runs/code/*code24*log*.json"):
    for f in glob.glob(pat):
        d = json.load(open(f))
        print("  file:", f, "keys:", list(d.keys())[:8])
        h = d.get("history") or []
        if h:
            print("   args:", {k: d.get("args", {}).get(k) for k in ("d", "window", "epochs")})
            print("   last:", h[-1])
        break
PY
echo "=== MEDICAL: audit + ship:"
grep -rE "test|acc|honest" lens/runs/vl/*.json 2>/dev/null | head -4
ls -la lens/runs/vl/*.npz 2>/dev/null | awk '{print $5, $9}' | head -4
echo "=== IMAGE: class-ship test accuracy (source of the 84.36 figure):"
grep -rE "84\.36|test_acc|test acc" lens/runs/image/*.json lens/runs/seetalk/*.json 2>/dev/null | head -4