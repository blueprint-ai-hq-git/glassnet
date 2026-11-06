#!/bin/bash
# Which structure makes arithmetic learnable? Plain script file (inline env-prefixed commands are blocked).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/SLOT-BINDING-PROBE.txt
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/slot_binding_probe.py > "$OUT" 2>&1
echo "exit=$?"
cat "$OUT"