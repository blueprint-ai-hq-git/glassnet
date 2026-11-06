#!/bin/bash
# run_g2_gate_test.sh — prove the concept-set gate works BOTH ways before the sweep.
#
# The first G2 sweep was void because T2_CONCEPT_SET was unset, so the models trained on the
# single-word concept set while the audit compared the 9 meaningful targets. The gate must:
#   (a) REFUSE to run when the env is unset (what happened last time)
#   (b) PROCEED when T2_CONCEPT_SET=meaningful_ext and T2_N_TAUGHT=9
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
PY=".venv-gpu/bin/python3"

echo "### (a) WITHOUT the env -- the gate must REFUSE"
env -u T2_CONCEPT_SET -u T2_N_TAUGHT G2_RUNGS=32 G2_SEEDS=0 G2_EPOCHS=1 \
  nice -n 19 ionice -c3 "$PY" -u scripts/g2_scaling.py 2>&1 | grep -A 5 "REFUSING TO RUN" \
  || echo "  !! gate did NOT refuse -- this is a problem"

echo
echo "### (b) WITH the env -- the gate must PASS"
T2_CONCEPT_SET=meaningful_ext T2_N_TAUGHT=9 G2_RUNGS=32 G2_SEEDS=0 G2_EPOCHS=1 \
  nice -n 19 ionice -c3 "$PY" -u scripts/g2_scaling.py 2>&1 | grep -E "concept set gate|^  D=" | head -4 \
  || echo "  !! gate did not pass -- this is a problem"