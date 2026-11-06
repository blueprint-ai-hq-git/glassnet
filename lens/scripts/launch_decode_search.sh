#!/bin/bash
# LAUNCHER for the decode-search measurement (2026-10-22 20:3x self-wake). A FILE, not an inline command.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-search-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=3G -p MemorySwapMax=1G \
  -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/run_decode_search.sh' \
  >> lens/runs/DECODE-SEARCH-LAUNCH.txt 2>&1 < /dev/null &
sleep 40
echo "$(date '+%F %H:%M:%S') unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)" \
  >> lens/runs/DECODE-SEARCH-LAUNCH.txt
pgrep -af decode_search_test >> lens/runs/DECODE-SEARCH-LAUNCH.txt 2>&1
tail -3 lens/runs/DECODE-SEARCH-RUN.txt >> lens/runs/DECODE-SEARCH-LAUNCH.txt 2>&1
