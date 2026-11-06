#!/bin/bash
# WALL-AB REPORT: wait for the A/B to finish, then turn the log into a table and grade P1-P5.
#
# Why a separate step: the A/B arms are still training under their own supervisor
# (lens/scripts/run_wall_ab_resume2.sh). This script only reads files - no GPU, no trainer, no lock -
# so it cannot collide with the run it is waiting for.
#
# Outputs:
#   lens/runs/WALL-AB-RESULTS.tsv   one row per finished arm, with the log line it came from
#   lens/runs/WALL-AB-GRADE.txt     P1-P5 graded by lens/scripts/wall_ab_grade.py (UNDETERMINED if an
#                                   arm is missing - it never guesses)
#   lens/runs/WALL-AB-REPORT.txt    the transcript of this step
#
# The predictions were written before any arm ran (docs/history/WALL-AB-PREDICTIONS.md); this script
# imports no threshold of its own.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=1

SRC=lens/runs/WALL-AB.txt
REP=lens/runs/WALL-AB-REPORT.txt
LIMIT_MIN=${1:-120}

say() { printf '%s %s\n' "$(date '+%F %H:%M:%S')" "$*" >> "$REP"; }

say "===== WALL-AB REPORT start (pid $$, wait limit ${LIMIT_MIN} min)"
if [ ! -f "$SRC" ]; then
  say "ABORT: $SRC missing"
  exit 2
fi

waited=0
while ! grep -q '^===== WALL-AB DONE' "$SRC"; do
  if [ "$waited" -ge "$LIMIT_MIN" ]; then
    say "TIMEOUT after ${waited} min with no 'WALL-AB DONE' line; reporting what exists (no verdict will be invented)"
    break
  fi
  sleep 60
  waited=$((waited + 1))
done

if grep -q '^===== WALL-AB DONE' "$SRC"; then
  say "DONE marker found after ${waited} min: $(grep '^===== WALL-AB DONE' "$SRC" | tail -1)"
else
  say "no DONE marker; partial report only"
fi

say "----- arm table"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 lens/scripts/wall_ab_results.py >> "$REP" 2>&1
say "results rc=$?"

say "----- grades"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 lens/scripts/wall_ab_grade.py > lens/runs/WALL-AB-GRADE.txt 2>&1
say "grade rc=$? -> lens/runs/WALL-AB-GRADE.txt"
tail -n 40 lens/runs/WALL-AB-GRADE.txt >> "$REP"

arms=$(grep -c $'^wall_' lens/runs/WALL-AB-RESULTS.tsv 2>/dev/null || echo 0)
say "===== WALL-AB REPORT end: $arms arm(s) in the table, 6 expected"
exit 0
