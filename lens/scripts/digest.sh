#!/bin/bash
# digest.sh - status with SUBSTANCE, not "it is running".
#
# The user's complaint (2026-10-19): "you have just set a cron job that will update me that doesn't help me
# anything because you just literally tell me it is running but what". So this leads with the numbers that
# answer "what is happening": which epoch, what accuracy, which way it is moving, how fast, and whether the
# GPU is actually working. If nothing is training it says so in capitals instead of reporting health.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME

echo "GlassNet $(date '+%a %H:%M')"

RUNNING=$(pgrep -cf "[t]rain_chat")
GPU=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null)

if [ "${RUNNING:-0}" -eq 0 ]; then
  echo
  echo "*** NOTHING IS TRAINING RIGHT NOW ***"
  echo "  no train_chat process. If that is unexpected, the last epoch line below says where it stopped."
else
  echo
  echo "TRAINING NOW:"
  for f in lens/runs/CLEAN-TRAIN.txt lens/runs/GPU-NOW.txt; do
    [ -f "$f" ] || continue
    grep -E "^  ep " "$f" | tail -2 | sed 's/^/  /'
    .venv-gpu/bin/python3 - "$f" <<'PY'
import re, sys
lines = [l for l in open(sys.argv[1]) if l.startswith("  ep ")]
if len(lines) >= 2:
    def acc(l):
        m = re.search(r"MASKED ppl\s+[\d.]+\s+acc\s+([\d.]+)%", l)
        return float(m.group(1)) if m else None
    a, b = acc(lines[-2]), acc(lines[-1])
    sec = re.search(r"\((\d+)s\)", lines[-1])
    if a is not None and b is not None:
        d = b - a
        arrow = "UP" if d > 0.05 else ("DOWN" if d < -0.05 else "FLAT")
        print(f"  -> masked accuracy {a:.2f}% to {b:.2f}% ({d:+.2f} points, {arrow})"
              + (f", {sec.group(1)}s per epoch" if sec else ""))
PY
  done
  echo "  GPU now: ${GPU}"
fi

echo
echo "LATEST VERDICTS AND GATES (real lines from the result files):"
for f in lens/runs/CLEAN-TRAIN.txt lens/runs/HONEST-ACCURACY.txt lens/runs/LEAK-CHECK-MIX.txt \
         lens/runs/CLEAN-VAL-MIX.txt lens/runs/REPAIR-CTRL.txt lens/runs/SETTINGS-SEARCH.txt \
         lens/runs/SLOT-BINDING-PROBE.txt lens/runs/METHODS-BENCH.txt lens/runs/REWARD-HACKING-AUDIT.txt \
         lens/runs/SIZE-SWEEP.txt lens/runs/LADDER-CTRL.txt lens/runs/LADDER-BIG.txt; do
  [ -f "$f" ] || continue
  OUT=$(grep -E "VERDICT|assistant-masked|exact on 10\.\.19|BEST STRUCTURE|EXACT byte matches|reward rate|ABORT|READABILITY GATE|WORD SALAD" "$f" 2>/dev/null | tail -3)
  [ -n "$OUT" ] && { echo "  -- $(basename "$f")"; echo "$OUT" | sed 's/^/     /'; }
done

echo
echo "PROBLEMS:"
for d in prepared_mix_honest prepared_mix_ctrl_honest; do
  if [ -f "data/chat/$d/ids_v12000.npy" ]; then echo "  corpus $d present"
  else echo "  MISSING CORPUS $d"; fi
done
echo "  OOM kills logged: $(grep -c 'oom-kill' /var/log/syslog 2>/dev/null || echo 0) (jobs are memory-capped, so a repeat kills the job, not the desktop)"
echo "  disk free: $(df -h . | tail -1 | awk '{print $4}')   ram: $(free -g | awk 'NR==2{print $3"G/"$2"G"}')"
NEWEST=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1)
echo "  newest checkpoint: $(basename "${NEWEST:-none}") $([ -n "$NEWEST" ] && echo "($(( ($(date +%s) - $(date -r "$NEWEST" '+%s')) / 60 )) min old)")"
echo
echo "WATCHDOG (last actions, or 'nothing wrong'):"
tail -3 lens/runs/WATCHDOG.txt 2>/dev/null | sed 's/^/  /' || echo "  (no report yet)"