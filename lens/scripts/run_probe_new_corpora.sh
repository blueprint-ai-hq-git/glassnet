#!/bin/bash
# Layout of the newly downloaded corpora, so the big-mix builder reads real columns.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-data/bin/python3 -u - <<'PY'
import os, pyarrow.parquet as pq
for name in ("OpenAssistant__oasst2", "Open-Orca__OpenOrca", "Open-Orca__SlimOrca",
             "meta-math__MetaMathQA", "nvidia__OpenMathInstruct-2"):
    base = f"data/external/{name}"
    if not os.path.isdir(base):
        print(f"--- {name}: not present yet"); continue
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    print(f"--- {name}: {len(files)} parquet files")
    for p in sorted(files)[:2]:
        t = pq.read_table(p)
        print(f"    {os.path.relpath(p, base)} rows={t.num_rows} cols={t.column_names[:10]}")
        for c in t.column_names[:6]:
            v = t[c][0].as_py()
            print(f"       {c}: {str(v)[:120]}")
PY