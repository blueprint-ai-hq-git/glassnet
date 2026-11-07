#!/bin/bash
# CONTROL ARM: the 500k rung with our step rule but the gap + agreement gates OFF.
#
# WHY: the ladder flattened between 500k and 2M (20.81 -> 21.29, 0.24 pp/doubling) while the arms refuse ~2/3 of
# their late steps. This arm asks whether the ceiling is the data or the gates. Identical to run_datascale_500k_s0.sh
# except --no-gates (validated in lens/runs/GATE-PROBE.txt: gates fire without it, never fire with it).
# Predictions, written before the launch: docs/history/NOGATE-500K-PREDICTIONS.md
#   G1 gates-off VAL >= 21.81% (gated arm + 1.0 pp); 19.81-21.81 = gates are not the ceiling; <19.81 = gates help
#   G2 overfitting signature (train-val gap pair)  G3 no gate/agreement lines  G4 concepts >= 36
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-500K-NOGATE.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_500k_nogate_s0"

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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_500k_nogate_s0" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_500k_nogate_s0\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_500k_nogate_s0" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-500K start (predictions: docs/history/NOGATE-500K-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --method ours --no-gates --max-windows 500000 (control: gated twin is 20.81% at DATASCALE-500K.txt:392)  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 500000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 --no-gates >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

# G3: mechanical proof this arm tested what it claims
ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL CHECK: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0; arm is void otherwise)"
grep -m1 "METHOD:" "$LOG" | cut -c1-140 >> "$LOG.tmp"; cat "$LOG.tmp" >> "$LOG"; rm -f "$LOG.tmp"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  say "----- epoch 3 and final result (read from the log above)"
  grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== NOGATE-500K DONE $(date '+%F %H:%M')"
else
  say "===== NOGATE-500K INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
