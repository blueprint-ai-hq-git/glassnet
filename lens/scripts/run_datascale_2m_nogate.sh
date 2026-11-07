#!/bin/bash
# The 2M rung WITHOUT the gates: was the flat ceiling the data, or the gates refusing most late steps?
#
# 500k gated 20.81 -> 500k ungated 22.05 (+1.24 pp, 7.7x faster). The ladder that "flattened" (20.81 -> 21.29 for
# 4x data) was entirely gated, so this arm re-runs the top rung ungated.
# Predictions, written before the launch: docs/history/NOGATE-2M-PREDICTIONS.md
#   U1 VAL >= 21.79 (gated 2M + 0.5)   U2 21.5-23.0 vs the ungated 500k arm   U3 ppl < 149.14
#   U4 mechanical (0 gate lines, GATES OFF)   U5 concepts >= 36
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-2M-NOGATE.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_2m_nogate_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_2m_nogate" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_2m_nogate\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_2m_nogate" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-2M start (predictions: docs/history/NOGATE-2M-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --method ours --no-gates --max-windows 2000000 (gated twin 21.29 at DATASCALE-2M.txt:1142; ungated 500k 22.05)  $(date '+%H:%M:%S')"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 2000000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 --no-gates >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0)"
grep -m1 "METHOD:" "$LOG" | cut -c1-130 >> "$LOG"
grep -m1 "tokens per parameter" "$LOG" | cut -c1-110 >> "$LOG"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== NOGATE-2M DONE $(date '+%F %H:%M')"
else
  say "===== NOGATE-2M INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
