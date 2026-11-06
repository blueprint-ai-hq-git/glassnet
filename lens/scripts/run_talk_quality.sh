#!/bin/bash
# Talk-quality meter runner. Clears PYTHONPATH (Hermes leaks its own numpy into child venvs).
# Usage: bash lens/scripts/run_talk_quality.sh <ship.npz> [label] [corpus-dir]
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP="${1:?usage: run_talk_quality.sh <ship.npz> [label] [corpus]}"
LABEL="${2:-$(basename "${SHIP%.npz}")}"
export CHAT_DATA_DIR="${3:-$PWD/data/chat/prepared_ext}"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_quality.py "$SHIP" "$LABEL"
