#!/bin/bash
# verify_clusters.sh — the loss changed, so re-verify before running anything.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
export T3_BACKEND=cupy

echo "############ 1. rebuild the cluster file (now with ids + pos_weight) ############"
"$PY" scripts/t3_concept_clusters.py 2>&1 | tail -6

echo
echo "############ 2. GRADIENT CHECK on the changed loss (v1 single-word path) ############"
bash "/run/media/yahya/HARD DISK 4/GlassNet/scripts/aux_gradcheck.sh" 2>&1 | grep -E "RESULT|PASS|FAIL" | tail -8

echo
echo "############ 3. GRADIENT CHECK on the CLUSTER path ############"
T2_CONCEPT_SET=clusters T2_LAM_CONCEPT=1.0 bash "/run/media/yahya/HARD DISK 4/GlassNet/scripts/aux_gradcheck.sh" 2>&1 | grep -E "RESULT|PASS|FAIL" | tail -8