#!/bin/bash
# t3_baselines.sh — item (3) part 1: vocab-1500 data + count baselines.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
: > runs/t3/baselines.log
nice -n 19 ionice -c3 "$PY" scripts/t3_baselines.py >> runs/t3/baselines.log 2>&1
echo "rc=$?" >> runs/t3/baselines.log
cat runs/t3/baselines.log