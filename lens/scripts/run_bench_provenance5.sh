#!/bin/bash
# Full recursive extraction of test_acc across ALL epoch entries (no last-3 limit), per file, so
# every accuracy row in the bench can name its ship, datum and SOURCE FILE.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json, glob
def walk(o, f, hits, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("test_acc", "n_test", "val_acc") and isinstance(v, (int, float)):
                hits.append((path, k, v))
            walk(v, f, hits, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f, hits, f"{path}[{i}]")
for f in sorted(glob.glob("lens/runs/image/*.json")):
    hits = []
    try:
        walk(json.load(open(f)), f, hits)
    except Exception:
        continue
    ta = [h for h in hits if h[1] == "test_acc"]
    nt = [h for h in hits if h[1] == "n_test"]
    if ta:
        print(f"  {f.split('/')[-1]:<34} max test_acc {max(h[2] for h in ta)*100:6.2f}%  "
              f"n_test {nt[-1][2] if nt else '?'}  ({len(ta)} epoch entries)")
PY