#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemd-run --user --scope --unit=glassnet-repair -p MemoryMax=6G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_repair_ctrl.sh"
