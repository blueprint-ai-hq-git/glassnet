#!/bin/bash
# t3_stageA.sh — item (3) core comparison: K=hidden=128 (355,357 params each),
# readable vs matched normal, 3 seeds, on the GPU.
#
# Item (3) is a NEW comparison group, so it runs entirely on ONE backend (GPU),
# per the pre-registered backend-consistency rule. GPU correctness was verified:
# gradient gate 9/9 PASS on cupy, CPU<->GPU parity 100% argmax agreement.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=128
export T3_SEEDS=0,1,2
export T3_MODELS=readable,normal
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T3_LR=0.1
export T3_TAG=K128
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "=== item (3) stage A  $(date '+%F %T') ===" >> runs/t3/models.log
nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> runs/t3/models.log 2>&1
echo "rc=$?" >> runs/t3/models.log