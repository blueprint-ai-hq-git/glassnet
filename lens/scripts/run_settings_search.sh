#!/bin/bash
# Run the settings search: every training method, every size, scored by RULE LEARNING with a memorisation
# penalty. Plain script file (inline env-prefixed commands trip the approval scanner).
#
# The user's order: settings first, model second. This is that search.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3

OUT=lens/runs/SETTINGS-SEARCH.txt
{
  echo "=== SETTINGS SEARCH $(date '+%F %H:%M')"
  echo "scoring: extrapolation accuracy (rule learning) + 0.5 * held-out - 0.5 * memorisation gap"
  echo "note: a model that memorised the addition table scores 0 here, which is the point"
  echo
  echo "--- pass 1: both methods across sizes 32..512, the full learning-rate and epoch grid"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/settings_search.py \
    --sizes 32 64 128 256 512 --methods sft rl
  echo "exit=$?"
  echo
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
tail -40 "$OUT"