#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemd-run --user --scope --unit=glassnet-rest -p MemoryMax=5G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_corpus_then_rest.sh"
