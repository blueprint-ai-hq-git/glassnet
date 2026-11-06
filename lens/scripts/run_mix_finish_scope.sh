#!/bin/bash
# Run the finish steps in their own memory-capped scope (never inside the agent's cgroup again).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemd-run --user --scope --unit=glassnet-corpus2 -p MemoryMax=4G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_mix_finish.sh"
