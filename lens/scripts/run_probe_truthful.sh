#!/bin/bash
# truthful_qa (Apache-2.0) layout -> JSONL for the bench harness.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-data/bin/python3 -u - <<'PY'
import json, os, glob
base = "data/external/truthful_qa"
files = glob.glob(base + "/**/*.parquet", recursive=True)
print("  parquet files:", [os.path.relpath(f, base) for f in files][:6])
try:
    import pyarrow.parquet as pq
    for f in files[:3]:
        t = pq.read_table(f)
        print(f"  {os.path.relpath(f, base)}: rows {t.num_rows} cols {t.column_names[:6]}")
except Exception as e:
    print("  parquet read failed:", type(e).__name__, e)
PY