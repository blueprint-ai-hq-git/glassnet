#!/bin/bash
# combo_matrix.py runner — EVERY modality combination through the shipped joint runtime,
# on real files, recorded to lens/runs/combos/combo_matrix.json.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv-gpu/bin/python3 -u lens/scripts/combo_matrix.py 2>&1 | tee lens/runs/combos/combo_matrix.log