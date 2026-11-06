#!/bin/bash
# Exact parquet layout + columns for the downloaded benchmarks, so the harness reads real fields.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-data/bin/python3 -u - <<'PY'
import os, pyarrow.parquet as pq
for name in ("mmlu", "gsm8k", "mbpp", "humaneval"):
    base = f"data/external/{name}"
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    print(f"--- {name}: {len(files)} parquet files")
    for p in sorted(files)[:3]:
        t = pq.read_table(p)
        print(f"    {os.path.relpath(p, base)}  rows={t.num_rows}  cols={t.column_names}")
        if t.num_rows:
            d = {c: t[c][0].as_py() for c in t.column_names}
            for k, v in d.items():
                print(f"       {k}: {str(v)[:150]}")
PY