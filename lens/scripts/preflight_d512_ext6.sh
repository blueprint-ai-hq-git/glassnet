#!/bin/bash
# PRE-FLIGHT for run_tied_d512_ext6.sh (2026-10-24 10:1x self-wake). NOT an experiment: it checks the WIRING that
# the real arm depends on, at 0.02% of its size, on CPU, on cores 6,7 so it does not touch the critical-path arm
# running on cores 0,1.
#   1. the exact flag set parses and trains at d=512 with --epochs/--lr-decay/--ckpt-seconds (1 epoch, 3,000 windows)
#   2. the `read ... < <(python3 - <<PY)` block that names the BEST epoch works on a real history file
#   3. the history row carries val_asst_acc AND val_masked_acc (the two columns the chained decider prints)
# Writes only lens/runs/D512-EXT6-PREFLIGHT.txt (created below) and lens/runs/chat/pf_d512ext6_*.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
LOG=lens/runs/D512-EXT6-PREFLIGHT.txt
{
  echo "=== D512-EXT6 PRE-FLIGHT $(date '+%F %H:%M:%S') (predictions: none needed - wiring check, not a measurement)"
  echo "--- step 1: same flags as the arm, 1 epoch, 3,000 windows, CPU, cores 6,7"
  taskset -c 6,7 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 512 --window 32 --epochs 1 --batch 128 --lr 0.3 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device cpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 3000 --ckpt-seconds 1800 --out pf_d512ext6
  echo "step1 exit=$?"
  echo "--- step 2: the read block from the runner, verbatim except the tag"
  read -r BEST_EP BEST_ASST BEST_LEGACY < <(.venv-gpu/bin/python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("lens/runs/chat/pf_d512ext6_history.jsonl") if l.strip()]
b = max(rows, key=lambda r: r["val_asst_acc"])
print(b["epoch"], f'{b["val_asst_acc"]*100:.2f}', f'{b["val_masked_acc"]*100:.2f}')
PY
)
  echo "step2 read -> ep=[$BEST_EP] asst=[$BEST_ASST] legacy=[$BEST_LEGACY]  (all three must be non-empty)"
  if [ -n "$BEST_EP" ] && [ -n "$BEST_ASST" ] && [ -n "$BEST_LEGACY" ]; then
    echo "step2 VERDICT: PASS - the best-epoch block works on a real history file"
  else
    echo "step2 VERDICT: FAIL - the runner would die at that line"
  fi
  echo "--- step 3: history row keys and the checkpoint it wrote"
  .venv-gpu/bin/python3 - <<'PY'
import json
row = json.loads(open("lens/runs/chat/pf_d512ext6_history.jsonl").readline())
need = ["epoch", "val_asst_acc", "val_masked_acc", "train_acc"]
print("  keys present:", {k: (k in row) for k in need})
print("  row:", {k: row.get(k) for k in need})
PY
  ls -l lens/runs/chat/pf_d512ext6_best.npz lens/runs/chat/pf_d512ext6_history.jsonl 2>&1
  echo "=== PRE-FLIGHT DONE $(date '+%F %H:%M:%S')"
} >> "$LOG" 2>&1
