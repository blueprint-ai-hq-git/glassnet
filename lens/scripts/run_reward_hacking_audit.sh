#!/bin/bash
# REWARD HACKING AUDIT - run it as a plain script file (inline env-prefixed commands trip the scanner).
# Answers the user's question "what about the reward hijacking" with measurements:
#   surface 1  which candidates the exact verifier accepts (bypass surface)
#   surface 2  luck-hijack rate: share of accepted correct samples that were lucky tail draws
#   collapse   answer-distribution entropy before/after RL against the held-out accuracy change
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3

OUT=lens/runs/REWARD-HACKING-AUDIT.txt
{
  echo "=== REWARD HACKING AUDIT $(date '+%F %H:%M')"
  for S in 8 32; do
    echo
    echo "## samples=$S at d=128"
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/reward_hacking_audit.py \
      --d 128 --samples "$S" --rounds 3 --pretrain 300
    echo "exit=$?"
  done
  echo
  echo "=== DONE $(date '+%F %H:%M')"
} > "$OUT" 2>&1
tail -46 "$OUT"