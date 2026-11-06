#!/bin/bash
# run_meter06b.sh — re-run METER-06 with the corrected random baseline.
#
# The first METER-06 run reported 0/9 for EVERY arm, including the arms that pass 9/9 under the old
# meter. That was an arithmetic error in the metric: the random baseline for the JOINT rate is
# q * s, and the first version divided by q alone, which capped a perfect concept at exactly 1.0
# against a bar of 2.0. Every concept therefore failed by construction.
#
# The BAR is unchanged (2x random). Only the miscomputed baseline is fixed. The broken run's output
# is kept, not overwritten, so both are on record.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=== keeping the broken first run for the record ==="
cp lens/runs/meter06/rescore.json lens/runs/meter06/rescore_BROKEN_firstrun.json 2>/dev/null
echo "  saved as rescore_BROKEN_firstrun.json"

echo
echo "=== METER-06, corrected baseline ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/rescore_graded.py 2>&1 | tail -95

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md /home/yahya/ 2>/dev/null
echo "docs copied"
