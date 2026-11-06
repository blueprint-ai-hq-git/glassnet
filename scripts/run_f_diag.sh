#!/bin/bash
# run_f_diag.sh — measure WHY the concepts died: same student, generated vs real data,
# logging the aux concept loss and the live AUROC that my F trainer never recorded.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export D_EPOCHS=6
for src in generated real; do
  D_SOURCE="$src" D_RULE=all nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u \
    scripts/f_diag_concepts.py 2>&1 | sed -n '/WHY DID THE CONCEPTS DIE/,$p'
  echo
done