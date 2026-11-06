#!/bin/bash
# ==========================================================================================================
# GlassNet on/off switch.  usage:  glassnet.sh on | off | status
# Made because stopping has to work at ANY random moment (Yahya leaves at no fixed time) and must never
# lose work. Everything the trainer needs to resume is already on disk: a checkpoint per epoch plus an
# RNG sidecar. "off" also asks the trainer to save an INTERRUPT checkpoint first (SIGTERM handler), so an
# epoch in flight is not thrown away - at worst a few minutes of steps.
# ==========================================================================================================
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
CHAT=lens/runs/chat
count() { local n; n=$(pgrep -cf "$1" 2>/dev/null); echo "${n:-0}"; }
newest() { ls -t $CHAT/*.npz 2>/dev/null | head -1; }
case "$1" in
  off)
    echo "=== STOPPING $(date '+%F %H:%M:%S')"
    # 1. ask the trainers to save an interrupt checkpoint and exit (SIGTERM handler in train_chat.py)
    pkill -TERM -f "[t]rain_chat" 2>/dev/null
    for i in $(seq 1 20); do [ "$(count '[t]rain_chat')" = "0" ] && break; sleep 1; done
    # 2. stop runners and any unit we created
    pkill -TERM -f "[l]ens/scripts/run_" 2>/dev/null
    for u in $(systemctl --user list-units --all --no-pager 2>/dev/null | grep -oE 'glassnet-[a-z0-9-]+\.(service|scope)' | sort -u); do
      systemctl --user stop "$u" >/dev/null 2>&1
    done
    sleep 3
    pkill -KILL -f "[t]rain_chat" 2>/dev/null
    systemctl --user reset-failed >/dev/null 2>&1
    # 3. pause the two loops so nothing restarts behind us
    echo "PAUSE_CRONS" > lens/runs/.stopped
    # 4. prove the state
    echo "  trainers: $(count '[t]rain_chat')   runners: $(count 'lens/scripts/run_')   units: $(systemctl --user list-units --no-pager 2>/dev/null | grep -ci glassnet)"
    echo "  newest checkpoint: $(basename "$(newest)" 2>/dev/null)"
    echo "  checkpoints kept: $(ls $CHAT/*.npz 2>/dev/null | wc -l)   disk free: $(df -h . | tail -1 | awk '{print $4}')"
    echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader)"
    echo "=== STOPPED. Nothing was deleted. 'bash lens/glassnet.sh on' resumes from the newest checkpoint."
    ;;
  on)
    echo "=== STARTING $(date '+%F %H:%M:%S')"
    rm -f lens/runs/.stopped
    # resume the newest run if it was interrupted mid-flight, otherwise let the queue take over
    echo "  newest checkpoint: $(basename "$(newest)" 2>/dev/null)"
    echo "  queue: $(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null) job(s) waiting"
    echo "  the two loops must be resumed for the queue to advance:"
    echo "    watchdog  83c757488676  +  self-wake 54c286b5a5eb   (resume them from the agent, or cronjob_manage)"
    echo "  to continue the interrupted run by hand:"
    echo "    bash lens/scripts/run_continue_newest.sh"
    ;;
  status)
    echo "=== GLASSNET STATUS $(date '+%F %H:%M:%S')"
    echo "  trainers: $(count '[t]rain_chat')   runners: $(count 'lens/scripts/run_')"
    ps -eo etime,args 2>/dev/null | grep '[t]rain_chat' | cut -c1-110
    echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader)"
    echo "  newest checkpoint: $(basename "$(newest)" 2>/dev/null)  ($(ls $CHAT/*.npz 2>/dev/null | wc -l) kept)"
    echo "  newest result files:"; ls -t lens/runs/*.txt 2>/dev/null | head -3 | sed 's/^/    /'
    echo "  queue: $(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null) job(s)"
    ;;
  *) echo "usage: glassnet.sh on | off | status"; exit 2 ;;
esac
