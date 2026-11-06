#!/bin/bash
# t3_smoke.sh — 2-epoch smoke test: does the item-(3) runner work, and how
# long is an epoch on the redesigned (vocab 1501) task?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_K=128
export T3_SEEDS=0
export T3_MODELS=readable,normal
export T3_EPOCHS=2
export T3_MIN_EPOCH=2
export T3_TAG=smoke
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
nice -n 19 ionice -c3 "$PY" scripts/t3_models.py 2>&1 | tail -30