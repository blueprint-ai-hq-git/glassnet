#!/bin/bash
# DATASCALE-HONEST: 10x the windows, on the honest mix, one arm, seed 0 - a script file, not an inline command.
#
# Predictions were written BEFORE the launch: docs/history/DATASCALE-PREDICTIONS.md
#   D1  200k windows lifts VAL acc from 12.18% to 15.18-20.18% (+3 to +8 pp)   [falsifiers both ways]
#   D2  final VAL ppl below 300 (baseline 399.50)
#   D3  one seed, 1.03 pp measured run-to-run spread, no significance claim
#
# WHY A SCRIPT, when a 200k arm was already run at 19:05 by other means: that run used a different corpus
# (13.3M cached tokens vs 294.2M), a different trainer method (plain SGD, gates OFF vs ours), a different
# val set (13,651 vs 87,035 windows) and a different concept set (40 kept vs 36), so its 13.12% cannot be
# compared to anything (docs/history/UNSCHEDULED-DATASCALE-RUN.md). This one changes exactly one flag from
# the measured 20k baseline: --max-windows 20000 -> 200000.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-HONEST.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_honest_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ -e "$SAVED/${TAG}_ep3.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: $TAG already has an epoch-3 checkpoint and a save line; nothing to do"
  exit 0
fi
if trainer_alive; then
  say "===== BLOCKED: a real train_chat.py run is alive; not competing with it"
  trainer_alive_print | cut -c1-140 >> "$LOG"
  exit 3
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
free_mb=${free_mb:-0}
if [ "$free_mb" -lt 3000 ]; then
  say "===== BLOCKED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)"
  exit 3
fi
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_honest" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_honest\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_honest" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== DATASCALE-HONEST start (predictions: docs/history/DATASCALE-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- command: identical to run_wall_ab.sh 'off' seed 0 except --max-windows 20000 -> 200000"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 200000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  say "----- result lines (read from the log above, this run only)"
  grep -E "^  ep 3/3|^  final VAL|NEW BEST" "$LOG" | tail -4 >> "$LOG"
  say "===== DATASCALE-HONEST DONE $(date '+%F %H:%M')"
else
  say "===== DATASCALE-HONEST INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
