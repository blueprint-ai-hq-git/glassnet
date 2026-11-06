#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
date '+NOW: %F %H:%M %Z'
echo
echo "=== GPU (frozen/hung check first) ==="
timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1 || echo "  nvidia-smi FAILED or TIMED OUT (possible hang)"
echo
echo "=== what is running ==="
pgrep -af "python3|ladder|train_" 2>/dev/null | grep -v grep | grep -v hermes | cut -c1-100 || echo "  no training processes"
echo
echo "=== recent crashes / OOM / kills in the kernel log ==="
dmesg 2>/dev/null | tail -5 || echo "  dmesg needs sudo (skipped)"
journalctl -k -n 8 --no-pager 2>/dev/null | tail -8 || echo "  journalctl unavailable"
echo
echo "=== any python tracebacks in the newest run logs ==="
for f in $(ls -t lens/runs/*/*.log 2>/dev/null | head -3); do
  echo "  --- $f ---"
  grep -iE "error|traceback|killed|oom|exception" "$f" 2>/dev/null | tail -4 || echo "    clean"
done
echo
echo "=== disk space (a full disk kills runs) ==="
df -h "/run/media/yahya/HARD DISK 4" /home/yahya 2>/dev/null | tail -3
echo
echo "=== memory now ==="
free -g | head -2
echo
echo "=== newest files written (proof of what last succeeded) ==="
ls -lt --time-style=+%H:%M lens/runs/ladder/*.json lens/runs/*/*.json 2>/dev/null | head -6
