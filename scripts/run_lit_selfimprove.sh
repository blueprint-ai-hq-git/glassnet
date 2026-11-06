#!/bin/bash
# run_lit_selfimprove.sh — verify the self-improvement literature via the arXiv API.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/lit_selfimprove.py 2>&1 | tail -60