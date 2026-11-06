#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-rebuild -p MemoryMax=9G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/rebuild_ctrl_corpus.sh"
