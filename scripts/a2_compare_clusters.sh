#!/bin/bash
# a2_compare_clusters.sh — is the k=16 regression a real change or just metadata?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
"$PY" - <<'EOF'
import json
a = json.load(open("/tmp/cc_v2_before.json"))   # the v2 file as it was
b = json.load(open("runs/t3/concept_clusters.json"))  # just regenerated with defaults

same_clusters = a["clusters"] == b["clusters"]
same_targets = a["targets"] == b["targets"]
same_k = a["k"] == b["k"]
same_above = a["n_above_floor"] == b["n_above_floor"]
print(f"  clusters identical      : {same_clusters}")
print(f"  targets identical       : {same_targets}")
print(f"  k identical             : {same_k}")
print(f"  n_above_floor identical : {same_above}")
print(f"  'method' string a = {a['method']!r}")
print(f"  'method' string b = {b['method']!r}")
only_meta = same_clusters and same_targets and same_k and same_above
print(f"\n  => difference is METADATA ONLY: {only_meta}")
if not only_meta:
    for i, (x, y) in enumerate(zip(a["clusters"], b["clusters"])):
        if x != y:
            print(f"  cluster {i} CHANGED: {x} -> {y}")
EOF