#!/bin/bash
# game_guard.sh — free the GPU the moment the owner starts a game.
#
# The owner's rule: "GPU jobs must EXIT for his game." gpu_watch.sh only logs VRAM; it does not
# stop anything. This does.
#
# Runs as a watchdog alongside any long job. Checks every 15 seconds for a game process and, if one
# appears, kills the training immediately -- training is resumable from its per-epoch checkpoints,
# a lost gaming session is not.
#
# Usage: bash game_guard.sh <pid-to-watch> [<pid> ...]
# Exits when all watched pids are gone.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG="lens/runs/game_guard.log"
WATCH="$*"

game_running () {
  pgrep -i -f "sober|roblox|RobloxPlayerBeta|cs2|hl2_linux|counter-strike|wine.*roblox" > /dev/null 2>&1
}

echo "$(date '+%F %T') guard started, watching pids: $WATCH" >> "$LOG"

while true; do
  alive=0
  for p in $WATCH; do
    if kill -0 "$p" 2>/dev/null; then alive=1; fi
  done
  if [ "$alive" -eq 0 ]; then
    echo "$(date '+%F %T') all watched jobs finished; guard exiting" >> "$LOG"
    exit 0
  fi
  if game_running; then
    echo "$(date '+%F %T') GAME DETECTED -- killing training to free the GPU" >> "$LOG"
    pkill -f "train_chat.py" 2>/dev/null
    pkill -f "train_chat_char.py" 2>/dev/null
    pkill -f "run_distill.py" 2>/dev/null
    pkill -f "run_blocker.py" 2>/dev/null
    pkill -f "run_optimise.sh" 2>/dev/null
    pkill -f "run_chat_" 2>/dev/null
    sleep 5
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOG" 2>/dev/null
    echo "$(date '+%F %T') training killed; GPU freed. Checkpoints are per-epoch and safe." >> "$LOG"
    exit 0
  fi
  sleep 15
done
