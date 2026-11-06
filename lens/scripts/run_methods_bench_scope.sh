#!/bin/bash
# Launch the method bench in its own systemd scope with a memory cap.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemd-run --user --scope --unit=glassnet-methods -p MemoryMax=5G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_methods_bench.sh"
