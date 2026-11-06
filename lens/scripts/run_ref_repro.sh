#!/bin/bash
# REPRODUCIBILITY of the 20k reference arm, one seed, one arm.
#
# WHY THIS AND NOT SOMETHING BIGGER: the wall A/B's control (wall_on) scored VAL 10.79% at seed 0 while
# the stored reference lc_w20000_s0 scored VAL 11.32% (LEARNCURVE.txt:4773) - a 0.53 pp gap under the SAME
# command and the SAME config header (docs/history/WALL-AB-P1-METRIC.md). The project's measured
# run-to-run spread on this path is 0.01-0.02 pp (docs/history/D512-REPRODUCIBILITY.md), so 0.53 pp does
# not look like noise and something is unexplained. Before any delta from the A/B is treated as a number,
# the reference has to be shown to reproduce. Nothing is re-decided here; this only measures.
#
# PREDICTIONS, written before the run (so the outcome cannot be read backwards):
#   P1. If the reference is reproducible, the fresh run lands VAL acc 11.3 +/- 0.1 (i.e. 11.2-11.4) and
#       ASST 11.96 +/- 0.1. Then the 0.53 pp gap is real and belongs to wall_on's block, not to noise.
#   P2. If instead the fresh run lands VAL 10.8 +/- 0.1, the stored 11.32 does not reproduce, and the band
#       P1 was graded against was built on a value this hardware cannot repeat.
#   P3. Neither outcome tells us which reading of P1 (metric-consistent vs as coded) is right. That stays
#       Yahya's decision; this run is evidence for it, not a substitute for it.
#   FALSIFIER for the whole test: if the command line differs from LEARNCURVE's 20k arm in any flag, the
#   run is void and must be reported as void, not pooled.
#
# Cost: one arm, 3 epochs, 20k windows, ~7 min on the GPU, ~400 MiB VRAM. nice 19, ionice c3.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG="lens/runs/REF-REPRO.txt"
CLAIM="lens/runs/GPU-CLAIM"
TAG="ref_repro_s0"

# one copy at a time, whatever wakes up alongside us
exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing done" | tee -a "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh

{
  echo "=== $(date '+%F %T') reference reproducibility, tag $TAG"
  if trainer_alive; then
    echo "BLOCKED: a real train_chat.py run is alive; not competing with it."
    trainer_alive_print
    exit 3
  fi
  if [ -e "$CLAIM" ] && ! grep -q "^owner: run_ref_repro" "$CLAIM" 2>/dev/null; then
    echo "BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM"); not starting a second GPU job."
    exit 3
  fi
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free_mb=${free_mb:-0}
  if [ "$free_mb" -lt 3000 ]; then
    echo "BLOCKED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)"
    exit 3
  fi

  printf 'owner: run_ref_repro\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
  trap 'rm -f "$CLAIM"' EXIT INT TERM

  echo "--- command (must match LEARNCURVE.txt:19-22 for SZ=20000, only --out differs)"
  echo "train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 --out $TAG --method ours --opt-lr 1.0  vram_free=${free_mb}MiB"

  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 \
    --out "$TAG" --method ours --opt-lr 1.0 2>&1 | tee "lens/runs/REF-REPRO-run.txt"
  rc=${PIPESTATUS[0]}
  echo "  exit=$rc"
  echo "--- the two columns, side by side with the reference (all read from log lines)"
  echo "  reference lc_w20000_s0  VAL 11.32%  ASST 11.96%   (LEARNCURVE.txt:4773)"
  echo "  wall_on seed 0          VAL 10.79%  ASST 11.42%   (WALL-AB.txt:392)"
  grep -E "^  ep 3/3|^  final VAL" lens/runs/REF-REPRO-run.txt | tail -2
  echo "DONE $(date '+%F %T') rc=$rc"
} 2>&1 | tee -a "$LOG"
