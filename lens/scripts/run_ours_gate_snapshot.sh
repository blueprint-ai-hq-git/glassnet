#!/bin/bash
# Read the live per-step gate logs of the ours-method run and write a snapshot. Read-only on the logs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 lens/scripts/ours_gate_snapshot.py
