#!/bin/bash
# 9G cap: the 4G cap is what killed the control-token step last night; 9G leaves the desktop room and still
# means an overreach kills this job instead of the machine.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-morning -p MemoryMax=9G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_morning_chain.sh"
