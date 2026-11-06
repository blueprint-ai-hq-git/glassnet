#!/bin/bash
# run_verify_claims.sh — claim-vs-artifact check.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
"$PY" scripts/verify_claims.py 2>&1