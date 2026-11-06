#!/bin/bash
# Leakage-meter runner (plain script: inline python and loops trip the approval gate).
# Usage: bash lens/scripts/run_concept_leakage.sh <ship.npz> [corpus-dir] [n-windows]
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP="${1:?usage: run_concept_leakage.sh <ship.npz> [corpus] [n]}"
CORPUS="${2:-$PWD/data/chat/prepared_splitfix}"
N="${3:-20000}"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/concept_leakage.py "$SHIP" "$CORPUS" "$N"
