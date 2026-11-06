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
  local attempt rc
  for attempt in 1 2; do
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
    say "----- $NAME seed $SEED attempt $attempt/2 : $*  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"
    CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
      .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
      --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed "$SEED" --device gpu --assistant-only \
      --train-turn-end --max-windows 20000 --gate-tol 0.05 \
      --out "${TAGP}_${NAME}_s${SEED}" --method ours --opt-lr 1.0 "$@" >> "$LOG" 2>&1
    rc=$?
    say "  $NAME seed $SEED exit=$rc"
    if [ "$rc" -eq 0 ]; then
      return 0
    fi
    # A KILLED ARM IS NOT A RESULT (2026-11-06 18:20, MEASURED): the first off/seed-2 trainer came back
    # exit=143 mid-epoch-1 and the first version of this script simply carried on to the next arm, which
    # would have published a three-arm block as if it were four. Retry once, then report it as missing.
    say "  $NAME seed $SEED attempt $attempt FAILED rc=$rc - retrying once (a killed arm is NOT a result)"
    sleep 5
  done
  say "-----$NAME seed $SEED GAVE UP after 2 attempts (last rc=$rc); it will be counted as MISSING"
  return 4
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

done_n=0
for S in 2 3; do for A in off on; do arm_done "$A" "$S" && done_n=$((done_n+1)); done; done
say "----- result lines in this log (each one is a finished arm's final VAL)"
grep -E "^  final VAL" "$LOG" | tail -4 >> "$LOG"
if [ "$done_n" -eq 4 ]; then
  say "===== WALL-REP DONE $(date '+%F %H:%M') arms_done=4 of 4"
else
  say "===== WALL-REP INCOMPLETE $(date '+%F %H:%M') arms_done=$done_n of 4 - missing arms are NOT results"
fi
exit 0
