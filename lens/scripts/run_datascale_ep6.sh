#!/bin/bash
# DATASCALE-EP6: the 200k arm with 6 epochs instead of 3, seed 0. One knob.
#
# WHY: seed 1's 3-epoch run was still improving at epoch 3 (VAL 16.24 -> 19.22 -> 19.71, ppl still falling)
# while seed 0 had flattened (17.48 -> 17.90 -> 17.43). If the schedule truncates the gain, every 3-epoch
# 200k number understates what data scale buys; if not, the extra epochs are wasted compute.
# Predictions, written before the launch: docs/history/DATASCALE-EP6-PREDICTIONS.md
#   E1 final VAL > 18.43% (+1.0 pp over the 3-epoch 17.43)   E2 final ppl < 207.29
#   E3 17.43-18.43 counts as a partial result, not E1     E4 never pooled with the 3-epoch arms
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-EP6.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_ep6_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ -e "$SAVED/${TAG}_ep6.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: $TAG already has an epoch-6 checkpoint and a save line; nothing to do"
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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_ep6" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_ep6\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_ep6" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== DATASCALE-EP6 start (predictions: docs/history/DATASCALE-EP6-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --max-windows 200000 --epochs 6 (one knob vs the 3-epoch 200k arm)  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 6 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 200000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep6.npz" ]; then
  say "----- all epoch lines and the final result (read from the log above)"
  grep -E "^  ep [0-9]+/6|^  final VAL" "$LOG" | tail -7 >> "$LOG"
  say "===== DATASCALE-EP6 DONE $(date '+%F %H:%M')"
else
  say "===== DATASCALE-EP6 INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
