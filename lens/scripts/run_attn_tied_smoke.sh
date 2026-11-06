#!/bin/bash
# P79: THE ATTENTION SMOKE — does attention=True + direct_mode='tied' actually TRAIN at real-data scale?
# Predictions written BEFORE this file was queued: lens/runs/ATTN-TIED-SMOKE-PREDICTIONS-P79.txt (P79a-f).
#
# TWO ARMS, ONE QUEUE LINE, BACK TO BACK: the watchdog starts one queue line at a time, so wrapping both arms in
# this one script is what keeps two trainers off the GPU (the OOM that killed this machine's runs on 2026-10-22).
#   ARM A = --attention ON      (out attn_tied_smoke_1)
#   ARM B = matched control     (out notied_smoke_1, byte-identical flags minus --attention)
# Same corpus, same 200,000 windows, same seed 0, same step rule (the OLD loop, --method standard), same mask.
# NOT A SHIP: 200k windows / 1 epoch is a wiring test. The decider is NOT run on it (P79e).
#
# WHAT IS ALREADY KNOWN AND IS NOT RE-TESTED HERE: lens/runs/ATTN-TIED-STRUCTURAL.txt (d=64) showed the
# combination builds, forwards and keeps the readout split exact at 2.73e-07, and that with attention ON the
# tied term must use the ATTENDED window (out["window_input"]) rather than emb[ctx].
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2

LOG=lens/runs/ATTN-TIED-SMOKE.txt
VRAM=lens/runs/_tmp/attn_smoke_vram.txt
mkdir -p lens/runs/_tmp

{
  echo "=== ATTENTION SMOKE (P79) $(date '+%F %H:%M') ==="
  echo "predictions: lens/runs/ATTN-TIED-SMOKE-PREDICTIONS-P79.txt (P79a-f)"
  echo "arm A = --attention, arm B = the same recipe without it; 200,000 windows, d=256, 1 epoch, seed 0, method standard"
  echo "structural check already done: lens/runs/ATTN-TIED-STRUCTURAL.txt (2.730e-07)"
  echo ""
} >> "$LOG" 2>&1

run_arm () {
  local TAG="$1"; shift
  local EXTRA="$*"
  : > "$VRAM"
  ( for i in $(seq 1 400); do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM" 2>/dev/null
      sleep 2
    done ) &
  local SAMPLER=$!
  local T0=$(date +%s)
  echo "--- ARM $TAG  start $(date '+%F %H:%M:%S')   flags: $EXTRA" >> "$LOG" 2>&1
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 1 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 200000 --method standard --out "$TAG" $EXTRA >> "$LOG" 2>&1
  local RC=$?
  kill $SAMPLER 2>/dev/null
  wait $SAMPLER 2>/dev/null
  local T1=$(date +%s)
  local PEAK=$(sort -n "$VRAM" 2>/dev/null | tail -1)
  echo "--- ARM $TAG  exit=$RC  wall=$((T1-T0))s  peak GPU memory seen during the arm: ${PEAK:-unmeasured} MiB" >> "$LOG" 2>&1
  echo "--- (scope MemoryPeak for this arm is read afterwards from the watchdog's own scope unit; see P79d)" >> "$LOG" 2>&1
  echo "" >> "$LOG"
  return $RC
}

run_arm attn_tied_smoke_1 --attention
RC_A=$?
run_arm notied_smoke_1
RC_B=$?

{
  echo "=== SMOKE SUMMARY $(date '+%F %H:%M') ==="
  echo "arm A (--attention) exit=$RC_A    arm B (control) exit=$RC_B"
  echo "read the two 'ep 1/1' lines above: P79b compares their ASST(by-position) accuracies (bar: within 3.0 points),"
  echo "P79c says the control lands in 12-22%, P79d the budget (wall <= 12 min/arm, peak GPU <= 2.5 GB)."
  echo "P79e: this is NOT put through the decider and NOT exported."
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> "$LOG" 2>&1
