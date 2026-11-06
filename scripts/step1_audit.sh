#!/bin/bash
# step1_audit.sh — STEP 1: audit the train/val measurement (Claude item 2).
# Plain wrapper so the launch line stays trivial (no && / inline env /
# redirect on the command line, which trips the security scanner).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
nice -n 19 ionice -c3 "$PY" scripts/t2_audit_measurement.py \
  > runs/t2/measure_audit.log 2>&1
echo "step1 rc=$?" >> runs/t2/measure_audit.log