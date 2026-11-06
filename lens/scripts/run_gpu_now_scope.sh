#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-gpu -p MemoryMax=6G -p CPUWeight=30 -p IOWeight=30 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_gpu_now.sh"
