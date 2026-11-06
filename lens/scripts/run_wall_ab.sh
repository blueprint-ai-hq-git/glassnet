#!/bin/bash
# THE CONCEPT WALL, MEASURED (question = docs/history/PROBLEMS.md OPEN #1).
# Predictions P1-P5 and the falsifiers are in docs/history/WALL-AB-PREDICTIONS.md, written BEFORE this ran.
#
# Three arms, one change each against each other, seeds 0 and 1, everything else matched to the stored
# 20k-window reference in lens/runs/CTX-ATTN.txt (final val 11.96/10.08, mean 11.02%):
#   wall_off : concepts only, NO bypass  (the concept wall with nothing beside it)
#   wall_on  : --direct --direct-mode tied            (+2,097,152 params, the current default)
#   wall_r64 : --direct --direct-mode lowrank -rank 64 (+1,292,416 params, PROBLEMS.md's "low-rank residual")
#
# GPU OWNERSHIP (2026-11-06 16:4x, MEASURED - AND THIS IS WHY THE QUEUED ATTEMPT WOULD HAVE STOOD NO CHANCE).
# This used to be a plain wait loop. watchdog.sh's resident keeper ("gpu_trainer" -> run_resume_best.sh) is
# re-launched by the 15-min tick whenever no train_chat.py is alive, and RESUME-BEST.txt shows it finishing the
# same 3-epoch run and being restarted over and over. Measured 16:37: a fresh trainer (pid 2896786) started 15 s
# after the previous one died, so "wait until the GPU is free" could not be satisfied for more than a second and
# the 2-hour abort would have fired having trained nothing. Two trainers on a 6 GB card is also the OOM class
# in BUGS.md. The fix touches no results: claim the card with lens/runs/GPU-CLAIM (watchdog.sh now honours that
# file in BOTH the resident guard and the queue launcher, so nothing else starts a trainer while we hold it),
# stop the resident unit GRACEFULLY (systemctl --user stop -> the trainer writes its interrupt checkpoint and
# exits cleanly; nothing is deleted), then wait for the card. On exit the claim is removed so his keeper policy
# resumes on the next tick.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/WALL-AB.txt
CLAIM=lens/runs/GPU-CLAIM
LOCK=lens/runs/GPU-WALLAB.lock

# ATOMIC: two chain agents woke within a minute of each other on 2026-11-06 16:5x, and both would have written
# the claim and run the 6 arms interleaved on one card. mkdir is atomic on this filesystem, so the loser exits
# without touching anything. Stale lock (owner pid gone) is taken over - never silently ignored.
if ! mkdir "$LOCK" 2>/dev/null; then
  old=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "===== REFUSED $(date '+%F %H:%M:%S'): $LOCK held by live pid $old" >> "$LOG"
    exit 0
  fi
  rm -rf "$LOCK" 2>/dev/null
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "===== REFUSED $(date '+%F %H:%M:%S'): could not take $LOCK" >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCK/pid"

release_claim() {
  [ -e "$CLAIM" ] && grep -q "^pid $$" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"
  [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"
}
trap release_claim EXIT

echo "===== WALL-AB START $(date '+%F %H:%M:%S')" >> "$LOG"
echo "pid $$ ; predictions in docs/history/WALL-AB-PREDICTIONS.md (written before this run, not edited since)" >> "$LOG"

echo "pid $$ run_wall_ab.sh wall_ab claimed $(date '+%F %H:%M:%S')" > "$CLAIM"
echo "===== GPU-CLAIM written (pid $$)" >> "$LOG"

# OWNERSHIP POLICY (rewritten 2026-11-06 16:57 after TWO measured aborts - the 15-min "give up" rule
# fired at 16:54:56 while the orphaned resident trainer was still on epoch 3 of 3, and had already
# fired once on 16:39. Nothing below deletes a result: the trainer's SIGTERM handler writes
# {out}_interrupt_ep{N}.npz + {out}_interrupt_rng.json and exits cleanly (train_chat.py:478-486),
# which is the SAME graceful path used at 16:53 when its own unit was stopped.)
#   1. hold the claim first (watchdog honours it, so no queue entry can start a second trainer);
#   2. stop any resident unit, and keep stopping it: the keeper relaunches on a 15-min tick;
#   3. wait up to 240 s for the card to free itself;
#   4. if a redundant resident trainer is still alive after 240 s, SIGTERM it ONCE (graceful, it
#      checkpoints itself), log the pids, and wait up to 180 s more;
#   5. start the arms only with >= 3000 MiB free, and abort (rc 3) rather than run two models at once.
stop_resident() {
  for u in $(systemctl --user list-units 'glassnet-res-gpu_trainer-*' --state=active --no-legend --plain 2>/dev/null | awk '{print $1}'); do
    systemctl --user stop "$u" 2>>"$LOG"
    echo "----- stopped resident unit $u (graceful SIGTERM; the trainer saves its own interrupt checkpoint)" >> "$LOG"
  done
}
stop_resident
# ONLY real python trainers are counted. `pgrep -f lens/scripts/train_chat` also matches any shell
# whose command line merely MENTIONS that path (an agent's own tool shell, or the pgrep inside this
# very loop), so the old test reported "a trainer is still alive" when none was: that is what aborted
# the 16:39 block at 16:54:56, and what made the 240 s graceful SIGTERM also hit pid 2996309, a shell.
trainer_pids() {
  for p in $(pgrep -f 'train_chat\.py' 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null)
    case "$exe" in *python3*) echo "$p" ;; esac
  done
}
WAITED=0
GRACEFUL=0
while [ -n "$(trainer_pids)" ]; do
  sleep 10; WAITED=$((WAITED+10))
  stop_resident
  if [ -e lens/runs/STOP-NIGHT ]; then echo "===== ABORT: STOP-NIGHT appeared while waiting" >> "$LOG"; exit 0; fi
  if [ "$WAITED" -ge 240 ] && [ "$GRACEFUL" -eq 0 ]; then
    PIDS=$(trainer_pids | tr '\n' ' ')
    echo "----- 240 s elapsed with a redundant resident trainer alive (pid $PIDS): ONE graceful SIGTERM so the A/B can run" >> "$LOG"
    for p in $PIDS; do kill -TERM "$p" 2>>"$LOG"; done
    GRACEFUL=1
  fi
  if [ "$WAITED" -ge 600 ]; then
    echo "===== ABORT: a trainer was still alive 10 min after the claim (and after one graceful SIGTERM); not starting a second one" >> "$LOG"
    exit 3
  fi
done
echo "===== GPU free after ${WAITED}s wait; starting arms" >> "$LOG"

run() { NAME=$1; SEED=$2; shift 2
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free_mb=${free_mb:-0}
  if [ "$free_mb" -lt 3000 ]; then
    echo "----- SKIP $NAME seed $SEED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)" >> "$LOG"
    return 0
  fi
  echo "----- $NAME seed $SEED : $*  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB" >> "$LOG"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --max-windows 20000 --gate-tol 0.05 \
    --out "wall_${NAME}_s$SEED" --method ours --opt-lr 1.0 "$@" >> "$LOG" 2>&1
  echo "  $NAME seed $SEED exit=$?" >> "$LOG"
}

for S in 0 1; do
  run off $S
  run on  $S --direct --direct-mode tied
  run r64 $S --direct --direct-mode lowrank --direct-rank 64
done
echo "===== WALL-AB DONE $(date '+%F %H:%M')" >> "$LOG"
