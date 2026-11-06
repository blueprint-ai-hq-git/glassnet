#!/bin/bash
# LAUNCHER for the context-dependence measurement (2026-10-22 20:3x self-wake). A FILE, not an inline command.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-ctxdep-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=3G -p MemorySwapMax=1G \
  -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/run_context_dependence.sh' \
  >> lens/runs/CONTEXT-DEPENDENCE-LAUNCH.txt 2>&1 < /dev/null &
sleep 25
echo "$(date '+%F %H:%M:%S') unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)" \
  >> lens/runs/CONTEXT-DEPENDENCE-LAUNCH.txt
pgrep -af context_dependence_test >> lens/runs/CONTEXT-DEPENDENCE-LAUNCH.txt 2>&1
tail -3 lens/runs/CONTEXT-DEPENDENCE-RUN.txt >> lens/runs/CONTEXT-DEPENDENCE-LAUNCH.txt 2>&1
