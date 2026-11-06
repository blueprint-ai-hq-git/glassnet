#!/bin/bash
# recon_rungs3.sh — Banking77 (loop iters 15-18) and PathMNIST artifacts.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== iter018 audit.md (Banking77) FULL ==="
cat runs/loop/iter018/audit.md 2>/dev/null | head -40
echo
echo "=== iter015 audit.md (Banking77) FULL ==="
cat runs/loop/iter015/audit.md 2>/dev/null | head -30
echo
echo "=== contents of a Banking77 iter dir ==="
ls -la runs/loop/iter018/ 2>/dev/null | head -20
echo
echo "=== any npz checkpoints under runs/loop ? ==="
find runs/loop -name "*.npz" 2>/dev/null | wc -l
find runs/loop -name "*.npz" 2>/dev/null | head -6
echo
echo "=== rung3 tuning dir ==="
ls -1 runs/rung3_normal_tune/ 2>/dev/null | head -12