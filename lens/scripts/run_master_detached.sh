#!/bin/bash
# Launch the master run in its OWN systemd scope with a memory cap.
#   * --scope puts it outside the hermes-gateway.service cgroup, so a gateway OOM cannot kill it
#   * MemoryMax=5G means that if it overreaches, the kernel kills THIS job instead of the whole desktop
#     (the machine has 14 GB total and the desktop plus browser take their share)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-master -p MemoryMax=5G -p CPUWeight=20 -p IOWeight=20 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_master.sh" &
sleep 5
systemctl --user status glassnet-master --no-pager 2>/dev/null | head -6
pgrep -af "run_master|run_mix_build|train_chat" | grep -v hermes | cut -c1-70
