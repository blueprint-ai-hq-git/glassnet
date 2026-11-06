#!/bin/bash
# why_killed.sh — evidence for why the re-audit keeps dying.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== memory ==="
free -h
echo
echo "=== OOM / kill messages in the kernel log (needs no sudo) ==="
dmesg 2>/dev/null | grep -iE "killed process|out of memory|oom" | tail -12
echo "(if empty, dmesg is restricted)"
echo
echo "=== journal OOM (last 20) ==="
journalctl --since "1 hour ago" 2>/dev/null | grep -iE "oom|killed process|reaudit" | tail -12
echo "(if empty, journal is restricted)"
echo
echo "=== supervisors / cron-like killers running ==="
ps -eo pid,etime,cmd | grep -iE "supervisor|gpu_watch|watchdog|kill" | grep -v grep | head -10
echo
echo "=== crontab ==="
crontab -l 2>/dev/null | head -12
echo
echo "=== partial results saved so far ==="
/home/yahya/.hermes/hermes-agent/venv/bin/python3 - <<'EOF'
import json, os
p = "runs/t3/reaudit_t2.json"
if os.path.exists(p):
    d = json.load(open(p))
    print(f"{len(d)} target(s) saved:")
    for k, v in d.items():
        print(f"   {k}: {v.get('status')}")
else:
    print("no reaudit_t2.json yet")
EOF