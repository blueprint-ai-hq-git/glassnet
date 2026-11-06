#!/bin/bash
# 9G cap: enough for the 294M-token rewrite (4G killed it), still below the machine's 14G so the desktop
# keeps breathing. If it ever overreaches, the kernel kills THIS job instead of the whole machine.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-finish -p MemoryMax=9G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_finish_and_train.sh"
