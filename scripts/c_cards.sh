#!/bin/bash
# c_cards.sh — write the Target C reader cards.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/c_cards.py 2>&1 | tail -6