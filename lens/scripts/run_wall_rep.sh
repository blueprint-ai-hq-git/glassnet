#!/bin/bash
# WALL-REP: two independent seeds (2, 3) of the two arms that decide the bypass question.
#
# Predictions and their falsifiers were written BEFORE this ran:
#   docs/history/WALL-REP-PREDICTIONS.md  (R1 direction replicates, R2 magnitude +1.0..+4.0 pp,
#                                          R3 spread <= 1.5 pp, R4 not pooled with the six-arm block)
# Why: the A/B's off-on deltas are +1.39 (s0) and +3.11 (s1) while three runs of one configuration span
# 1.03 pp (docs/history/REF-REPRO.md). One seed per arm cannot separate that. This block is a replication
# on fresh seeds and does not touch, re-grade or re-open WALL-AB.txt.
#
# Same trainer command as run_wall_ab.sh:113-117 and run_learncurve.sh:19-22; only --out and --seed differ.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/WALL-REP.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAGP="wall_rep"

# one copy at a time, even if several agents wake at once
exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

arm_done() {  # $1=name $2=seed : a save line in THIS log AND the epoch-3 npz on disk
  grep -q "saved to .*${TAGP}_$1_s$2_" "$LOG" 2>/dev/null || return 1
  ls "$SAVED/${TAGP}_$1_s$2"_ep*.npz >/dev/null 2>&1 || return 1
  return 0
}

run_arm() {  # $1=name $2=seed, rest passed to the trainer
  local NAME="$1" SEED="$2"; shift 2
  if arm_done "$NAME" "$SEED"; then
    say "----- SKIP $NAME seed $SEED: recorded in $LOG and its npz exists"
    return 0
  fi
  if trainer_alive; then
    say "----- SKIP $NAME seed $SEED: another real train_chat.py run is alive"
    trainer_alive_print | cut -c1-120 >> "$LOG"
    return 3
  fi
  local free_mb
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free_mb=${free_mb:-0}
  if [ "$free_mb" -lt 3000 ]; then
    say "----- SKIP $NAME seed $SEED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)"
    return 3
  fi
  if [ -e "$CLAIM" ] && ! grep -q "^owner: run_wall_rep" "$CLAIM" 2>/dev/null; then
    say "----- SKIP $NAME seed $SEED: GPU-CLAIM held by $(head -1 "$CLAIM")"
    return 3
  fi
  printf 'owner: run_wall_rep\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
  say "----- $NAME seed $SEED : $*  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed "$SEED" --device gpu --assistant-only \
    --train-turn-end --max-windows 20000 --gate-tol 0.05 \
    --out "${TAGP}_${NAME}_s${SEED}" --method ours --opt-lr 1.0 "$@" >> "$LOG" 2>&1
  say "  $NAME seed $SEED exit=$?"
  return 0
}

cleanup() {
  [ -e "$CLAIM" ] && grep -q "^owner: run_wall_rep" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"
}
trap cleanup EXIT INT TERM

say "===== WALL-REP start (predictions: docs/history/WALL-REP-PREDICTIONS.md, written before this run)"
for S in 2 3; do
  run_arm off "$S"
  run_arm on  "$S" --direct --direct-mode tied
done

say "----- arm table (VAL acc and ASST acc read from each epoch-3 line)"
grep -E "^  ep 3/3" "$LOG" | sed -E 's/.*VAL ppl ([0-9.]+) acc ([0-9.]+)%.*ASST\(by-position\) ppl [0-9.]+ acc ([0-9.]+)%.*/  VAL \2%  ASST \3%/' >> "$LOG"
say "===== WALL-REP DONE $(date '+%F %H:%M')"
grep -cE "^  final VAL" "$LOG" | sed 's/^/  arms finished: /' >> "$LOG"
exit 0
