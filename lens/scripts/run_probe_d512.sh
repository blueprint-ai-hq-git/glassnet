#!/bin/bash
# COST PROBE: the 37.37% headline's own configuration, one epoch, with our step rule and the gates off.
#
# The headline (TIED-D512-EXT6.txt) is d=512 / --direct --direct-mode tied / 16.9M windows / 6 epochs / lr 0.3 /
# plain SGD. This probe changes ONLY the step rule and the gates (--method ours --no-gates) and runs ONE epoch, to
# measure what an epoch costs - the 6-epoch attempt is only worth scheduling if an epoch is affordable.
# Predictions, written before the launch: docs/history/D512-COST-PROBE-PREDICTIONS.md
#   P1 one epoch 2,500-6,000 s   P2 VAL 28-36%   P3 mechanical   P4 the ext6 run and its checkpoints are untouched
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/PROBE-D512.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="probe_d512_relgrad"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ -e "$SAVED/${TAG}_ep1.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: $TAG already has an epoch-1 checkpoint and a save line; nothing to do"
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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_probe_d512" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_probe_d512\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_probe_d512" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== PROBE-D512 start (predictions: docs/history/D512-COST-PROBE-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- probe d512 : --method ours --no-gates --direct tied --d 512 --batch 1024 --lr 0.3 --epochs 1, FULL corpus  $(date '+%H:%M:%S')"

start=$(date '+%s')
CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 512 --window 32 --epochs 1 --batch 1024 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --direct --direct-mode tied \
  --out "$TAG" --method ours --no-gates >> "$LOG" 2>&1
rc=$?
end=$(date '+%s')
say "  exit=$rc  wall=$((end-start))s"

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0)"
grep -m1 "METHOD:" "$LOG" | cut -c1-130 >> "$LOG"
grep -m1 "STEP RULE:" "$LOG" | cut -c1-120 >> "$LOG"
grep -m1 "model: " "$LOG" | cut -c1-70 >> "$LOG"
grep -m1 "tokens per parameter" "$LOG" | cut -c1-110 >> "$LOG"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep1.npz" ]; then
  grep -E "^  ep 1/1|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== PROBE-D512 DONE $(date '+%F %H:%M')  epoch_wall=$((end-start))s"
else
  say "===== PROBE-D512 INCOMPLETE $(date '+%F %H:%M') rc=$rc"
fi
exit 0
