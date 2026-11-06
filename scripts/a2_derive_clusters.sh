#!/bin/bash
# a2_derive_clusters.sh — derive the A2 k=32 cluster targets.
# Step 1 is a regression check: with defaults (k=16) the code must still produce the SAME
# clusters/targets as the v2 file, proving the method was not altered. The check compares
# the DATA, not raw bytes -- the JSON carries a descriptive "method" string that now
# mentions k, and a byte comparison would flag that harmless difference.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a2_clusters.log

cp -f runs/t3/concept_clusters.json /tmp/cc_v2_before.json

echo "=== 1. regression: k=16 defaults must reproduce the v2 clusters/targets ===" | tee "$LOG"
"$PY" -u scripts/t3_concept_clusters.py 2>&1 | tail -2 | tee -a "$LOG"
"$PY" - <<'EOF' | tee -a "$LOG"
import json
a = json.load(open("/tmp/cc_v2_before.json"))
b = json.load(open("runs/t3/concept_clusters.json"))
ok = (a["clusters"] == b["clusters"] and a["targets"] == b["targets"]
      and a["k"] == b["k"] and a["n_above_floor"] == b["n_above_floor"])
print(f"    clusters identical: {a['clusters'] == b['clusters']} | "
      f"targets: {a['targets'] == b['targets']} | k: {a['k'] == b['k']}")
print("    REGRESSION OK (method unchanged)" if ok else "    !! REGRESSION FAILED")
raise SystemExit(0 if ok else 1)
EOF
RC=$?
[ "$RC" -ne 0 ] && { echo "!! regression failed" | tee -a "$LOG"; exit 1; }

echo "=== 2. derive the A2 k=32 set ===" | tee -a "$LOG"
T3C_K=32 T3C_OUT=concept_clusters32.json "$PY" -u scripts/t3_concept_clusters.py \
  2>&1 | tail -45 | tee -a "$LOG"

[ -f runs/t3/concept_clusters32.json ] || { echo "!! no k=32 output" | tee -a "$LOG"; exit 1; }
echo "=== done  $(date '+%T') ===" | tee -a "$LOG"