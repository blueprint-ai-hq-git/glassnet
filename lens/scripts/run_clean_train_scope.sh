#!/bin/bash
# Launch the clean-corpus training in its own systemd scope (own cgroup + memory cap, so a failure kills the
# job instead of the desktop, and a gateway restart cannot take it down).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-clean -p MemoryMax=6G -p CPUWeight=30 -p IOWeight=30 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_clean_train.sh"
