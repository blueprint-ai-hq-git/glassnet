#!/bin/bash
# run_f_diag_long.sh — the concepts trained FINE for 6 epochs (AUROC 0.64-0.71). The F students
# ran 40 epochs with patience 10. So the hypothesis is now: the concepts train and then get
# DESTROYED by prolonged training on data the model cannot fit. This watches 30 epochs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export D_EPOCHS=30
D_SOURCE=generated D_RULE=all nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u \
  scripts/f_diag_concepts.py 2>&1 | sed -n '/WHY DID THE CONCEPTS DIE/,$p' | tail -36