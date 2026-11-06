#!/bin/bash
# Rebuild the combined corpus in its OWN systemd scope with a memory cap, so that if it overreaches on
# memory the kernel kills THIS job and not the desktop or the agent (that is what happened at 17:22).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemd-run --user --scope --unit=glassnet-corpus -p MemoryMax=4G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_mix_build.sh"
