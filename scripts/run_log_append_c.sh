#!/bin/bash
# run_log_append_c.sh — append the Target C LOG entry.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/log_append_c.py 2>&1 | tail -4