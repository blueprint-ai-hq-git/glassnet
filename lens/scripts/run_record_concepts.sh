#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_concepts_test.py 2>&1 | tail -5
