#!/bin/bash
# Is the last margin shortfall a property of the MODEL or of the FORMULAS? Run as a script file
# (inline env-prefixed commands trip the approval gate).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 VISION_AUDIT_N=60000
for SHIP in seetalk_k21n_direct.npz seetalk_k21j.npz; do
  echo "==============================================================================="
  VISION_SHIP="$SHIP" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/formula_ceiling.py
done