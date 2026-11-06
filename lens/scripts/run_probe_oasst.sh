#!/bin/bash
# (a) is parquet reading available? (b) what is inside the pre-existing oasst1_trees.jsonl.gz --
# if a previous session already converted oasst1 into a usable shape, reuse it instead of
# re-deriving from parquet.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
for mod in ("pyarrow", "pandas", "datasets"):
    try:
        m = __import__(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {mod}: MISSING ({type(e).__name__})")
import gzip, json
p = "data/chat/oasst1_trees.jsonl.gz"
n = 0
with gzip.open(p, "rt") as f:
    for line in f:
        d = json.loads(line)
        print("  record keys:", list(d.keys())[:8])
        for k, v in d.items():
            if isinstance(v, str):
                print(f"    {k}: {v[:200]!r}")
            else:
                print(f"    {k}: {type(v).__name__} {str(v)[:120]}")
        n += 1
        if n >= 2:
            break
PY
echo "=== line count (approximate):"
zcat data/chat/oasst1_trees.jsonl.gz 2>/dev/null | wc -l
echo "=== orca_math / speech_commands / mnist download status:"
du -sh data/external/orca_math data/external/speech_commands data/external/mnist 2>/dev/null
tail -2 /tmp/download_external.log