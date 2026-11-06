#!/bin/bash
# TALK REGRADE through the pad-free decoder (CPU only, no training, no GPU).
# Question: were the "talk 0/6" verdicts on today's arms real, or produced by the id-0 left padding
# that PROMPT-ENCODING-PREDICTIONS.md measured? Predictions for THIS run:
# docs/history/TALK-REGRADE-PREDICTIONS.md (written before the script ran).
# Chain target: /home/yahya/.local/bin/glassnet_chain.sh lens/scripts/run_talk_regrade.sh talk_regrade
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/TALK-REGRADE.txt

echo "===== TALK-REGRADE START $(date '+%F %H:%M:%S')" >> "$LOG"
# cores 2,3 only: the GPU job (wall_ab) owns 0,1 and the card; this uses neither.
taskset -c 2,3 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/talk_regrade.py >> "$LOG" 2>&1
rc=$?
echo "===== TALK-REGRADE END $(date '+%F %H:%M:%S') rc=$rc" >> "$LOG"
exit $rc
