#!/bin/bash
# Audits every K21k ship against the SAME pinned datum (n=60000, seed 0) and prints one compact
# table, so the lambda comparison is a single row-per-ship read rather than three separate runs.
# The datum is the measuring instrument: its size and seed are pinned here on purpose.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 VISION_AUDIT_N=60000
for SHIP in "$@"; do
  echo "======================================================================================"
  VISION_SHIP="$SHIP" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/audit_strict_vision.py 2>&1 | grep -E "ship |weak|mislab|STRICT"
done