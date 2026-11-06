#!/bin/bash
# recon_rungs.sh — what audit artifacts exist for each earlier rung?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== runs/ dirs ==="
ls -d runs/*/ 2>/dev/null
echo
echo "=== files per run dir (audit/concept/gate) ==="
for d in runs/*/; do
  n=$(ls "$d" 2>/dev/null | grep -icE "audit|concept|gate")
  echo "$d -> $n"
done
echo
echo "=== audit/concept files anywhere under runs/ (maxdepth 3) ==="
find runs -maxdepth 3 \( -iname "*audit*" -o -iname "*concept*" \) 2>/dev/null | sort | head -40
echo
echo "=== top-level docs ==="
ls -1 *.md 2>/dev/null