#!/bin/bash
# ceiling.sh — diagnostic: contribution ceiling for the K=32 readable.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export CE_K=32
export CE_LR=1.0
export CE_SEED=0
export CE_TOPNAME=6
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
"$PY" scripts/t3_contrib_ceiling.py 2>&1 | tee -a runs/t3/ceiling.log