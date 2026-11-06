#!/bin/bash
# Persistent supervisor: a daemon, not a cron tick, so the machine never waits for a human to notice.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
systemctl --user reset-failed 2>/dev/null
systemd-run --user --scope --unit=glassnet-supervisor -p MemoryMax=1G bash \
  "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/supervisor.sh"
