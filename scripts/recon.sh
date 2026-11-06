#!/bin/bash
# recon.sh — morning recon after the shutdown. Plain script so the launch line
# stays trivial (no && / inline substitution -> no approval prompt).
echo "=== time ==="
date '+%F %T %Z (%A)'
echo "uptime: $(uptime -p 2>/dev/null || uptime)"
echo
echo "=== disk 4 ==="
df -h "/run/media/yahya/HARD DISK 4" 2>/dev/null | tail -1
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo
echo "=== gpu venv ==="
"/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3" -c "import cupy; print('cupy', cupy.__version__); print('devices', cupy.cuda.runtime.getDeviceCount())" 2>&1 | tail -3
echo
echo "=== cpu temp / load ==="
sensors 2>/dev/null | grep -iE "tctl|tccd|core 0|package" | head -3
cat /proc/loadavg
echo
echo "=== key files ==="
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
ls -la GATE-RULE-T2.md GLASSNET-RESUME-03.md LOG.md 2>&1 | awk '{print "  "$5" "$9}'
echo
echo "=== running training procs ==="
ps -eo pid,cmd | grep -E "[t]3_models|[t]3_audit|[o]vernight_supervisor" | head -5
echo "(none above = idle)"
