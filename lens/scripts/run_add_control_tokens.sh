#!/bin/bash
# Build the control-token corpus (CPU only; the GPU is busy with the size sweep).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=/tmp/ctrl_tokens.log
{
  echo "=== CONTROL TOKENS $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/add_control_tokens.py \
    data/chat/prepared_ext_honest data/chat/prepared_ctrl
  echo "exit=$?"
} > "$LOG" 2>&1
tail -14 "$LOG"