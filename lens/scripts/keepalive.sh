#!/bin/bash
# GLASSNET KEEP-ALIVE - runs on a schedule (no model, no token cost).
# Does the mechanical parts of "staying active": keeps the GPU fed, keeps the live page up, reports only
# when it acted or found a problem. It cannot think; it cannot read issues; it cannot decide. That is what
# scheduled agent runs are for, and they cost tokens.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 0
ACTED=""
GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)

# 1. GPU idle AND nothing training? refill and restart the queue.
if [ "$(pgrep -cf train_chat)" = "0" ] && [ "$(pgrep -cf gpu_queue_runner)" = "0" ]; then
  if [ ! -s lens/runs/GPU-QUEUE.txt ] || grep -q "QUEUE DRAINED" lens/runs/GPU-QUEUE-RUNS.txt 2>/dev/null; then
    cat > lens/runs/GPU-QUEUE.txt <<'Q'
bash lens/scripts/run_gpu_best.sh
bash lens/scripts/run_gpu_seed2.sh
bash lens/scripts/run_gpu_d512.sh
Q
  fi
  setsid nohup bash lens/scripts/gpu_queue_runner.sh >/dev/null 2>&1 &
  ACTED="$ACTED restarted-gpu-queue"
fi

# 2. live page down? restart it.
if ! curl -s -o /dev/null --max-time 4 http://127.0.0.1:8770/; then
  systemctl --user restart glassnet-liveview >/dev/null 2>&1 || \
    ( cd "/run/media/yahya/HARD DISK 4/GlassNet" && setsid nohup .venv-gpu/bin/python3 lens/liveview.py >> lens/runs/LIVEVIEW.txt 2>&1 & )
  ACTED="$ACTED restarted-liveview"
fi

# 3. disk nearly full? never silently.
FREE=$(df -h . | tail -1 | awk '{print $4}')
case "$FREE" in *G) [ "${FREE%G}" -lt 10 ] && ACTED="$ACTED DISK-LOW($FREE)";; esac

# speak only if something happened
if [ -n "$ACTED" ]; then
  echo "keepalive $(date '+%F %H:%M') GPU ${GPU}% -$ACTED"
  echo "keepalive $(date '+%F %H:%M') GPU ${GPU}% -$ACTED  (trainer $(pgrep -cf train_chat), live $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8770/))" >> lens/runs/KEEPALIVE.txt
fi
