#!/bin/bash
# run_meter06.sh — re-score existing checkpoints under the pre-registered METER-06. No retraining.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/rescore_graded.py 2>&1 | tail -95
cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"