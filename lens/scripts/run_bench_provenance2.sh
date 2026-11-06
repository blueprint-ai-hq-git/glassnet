#!/bin/bash
# Pin the exact provenance of the medical and image accuracy rows: which SHIP, which datum, which
# file the number comes from. The audit JSONs and run logs are the only sources accepted.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== MEDICAL ship candidates (med_cc3 line):"
ls -la lens/runs/vl/ | grep -E "med_cc3" | awk '{print $5, $9}' | head -6
for f in lens/runs/vl/med_cc3_log.json lens/runs/vl/med_cc3*log*.json; do
  [ -f "$f" ] && echo "  $f:" && tail -c 600 "$f"
done
echo "=== IMAGE: highest test_acc across image logs, with its ship + datum:"
.venv-gpu/bin/python3 -u - <<'PY'
import json, glob, os
best = None
for f in glob.glob("lens/runs/image/*log*.json") + glob.glob("lens/runs/seetalk/*log*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    h = d.get("history") or []
    cands = [r for r in h if isinstance(r, dict) and "test_acc" in r]
    if not cands:
        continue
    r = cands[-1]
    if best is None or r["test_acc"] > best[1]:
        best = (f, r["test_acc"], r.get("n_test"), d.get("args", {}))
print("  best recorded test_acc:", best[0] if best else "none")
if best:
    print(f"    test_acc {best[1]*100:.2f}%   n_test {best[2]}   args {best[3]}")
PY
echo "=== vision audit json (the 19/19 datum):"
head -c 400 lens/runs/seetalk/strict_vision_audit.json