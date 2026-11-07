#!/bin/bash
# Does the data lever keep paying past 500k? 2M windows, seed 0, d=256, batch 256, 3 epochs.
#
# LADDER AT SEED 0: 12.18% (20k) -> 17.43% (200k) -> 20.81% (500k), i.e. 1.58 then 2.56 pp per doubling.
# NOT comparable to the project's existing 2M result (24.20%) which used batch 1024 and a different sampling
# path - this run must never be quoted against it.
# Predictions, written before the launch: docs/history/DATASCALE-2M-PREDICTIONS.md
#   T1 VAL > 20.81%   T2 21.5-24.0%   T3 ppl < 171.47   T4 concepts >= 36
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-2M.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_2m_s0"

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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_2m_s0" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_2m_s0\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_2m_s0" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== DATASCALE-2M start (predictions: docs/history/DATASCALE-2M-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --max-windows 2000000 (4x the 500k arm, same batch 256 and 3 epochs)  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 2000000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  say "----- epoch 3 and final result (read from the log above)"
  grep -E "^  ep 3/3|^  final VAL|tokens per parameter" "$LOG" | tail -3 >> "$LOG"
  say "===== DATASCALE-2M DONE $(date '+%F %H:%M')"
else
  say "===== DATASCALE-2M INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
