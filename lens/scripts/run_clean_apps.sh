#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -22
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "lens/scripts")
import clean

# SECOND dataset, different shape: apps_train.jsonl is 107 MB of code-app descriptions with
# a different schema than magicoder (different field names, no instruct wrapper).
p = "data/code/apps_train.jsonl"
items = clean.load_any(p)
print(f"  load_any({p}): {len(items):,} items")
kept, rep = clean.clean(items, enabled=True, verbose=True)
json_out = rep
import json
os.makedirs("lens/runs/clean", exist_ok=True)
json.dump({"dataset": p, "n_in": len(items), "n_kept": len(kept),
           "dropped": rep["dropped"], "rule_detail": rep.get("rule_detail")},
          open("lens/runs/clean/apps_train_report.json", "w"), indent=1)
print(f"  -> {len(kept):,}/{len(items):,} = {len(kept)/max(len(items),1)*100:.2f}% kept; report saved")
PY
