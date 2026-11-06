#!/bin/bash
# recon_rung_code.sh — which scripts load the rung-1/2/3 models and data?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== scripts/ ==="
ls -1 scripts/ | head -60
echo
echo "=== scripts mentioning phase2 / loop / rung ==="
grep -rln "phase2\|runs/loop\|rung" scripts/*.py 2>/dev/null | head -12
echo
echo "=== glassnet package modules ==="
ls -1 glassnet/
echo
echo "=== who defines the loop audit (fidelity / leans toward)? ==="
grep -rln "leans toward\|fidelity" scripts/*.py glassnet/*.py 2>/dev/null | head -8