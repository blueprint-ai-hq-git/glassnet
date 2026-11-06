#!/bin/bash
# Exactly the numbers + datums for the medical and image bench rows (no truncation).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json, glob
d = json.load(open("lens/runs/vl/med_cc3_log.json"))
print("MEDICAL med_cc3_ep8 -- top-level keys:", list(d.keys())[:8])
def dig(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("test_acc", "n_test", "val_acc", "n_val") and not isinstance(v, (dict, list)):
                print(f"   {path}.{k} = {v}")
            else:
                dig(v, f"{path}.{k}")
    elif isinstance(o, list) and o and isinstance(o[-1], dict):
        dig(o[-1], f"{path}[-1]")
dig(d)
best = None
for f in glob.glob("lens/runs/image/*log*.json"):
    try:
        dd = json.load(open(f))
    except Exception:
        continue
    for r in (dd.get("history") or []):
        if isinstance(r, dict) and "test_acc" in r:
            if best is None or r["test_acc"] > best[1]:
                best = (f, r["test_acc"], r.get("n_test"))
print("IMAGE best recorded:", best)
PY