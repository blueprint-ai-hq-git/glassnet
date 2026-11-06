#!/bin/bash
# READ-ONLY inspection runner (see lens/scripts/_inspect_marker.py for what it answers and why).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
{
  echo "=== markers in the corpus $(date '+%F %H:%M') (read-only) ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/_inspect_marker.py
  echo "exit=$?"
} >> lens/runs/CORPUS-MARKER-INSPECT.txt 2>&1
