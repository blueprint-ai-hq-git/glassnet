#!/bin/bash
# run_tied_w64_b256_recovery.sh — CONTINGENCY ONLY, NOT QUEUED. Written 2026-10-25 11:1x by the self-wake.
#
# WHY THIS FILE EXISTS: lens/scripts/run_tied_w64.sh runs BOTH window-64 arms in one chained job. If ARM 2
# (ct_d256_w64_b256) is killed before it writes a best checkpoint, the chain stops at its own missing-best guard
# (run_tied_w64.sh lines 82-87, exit 7) and the DECIDER never runs — the arm is then unmeasured. Re-running
# run_tied_w64.sh in that state is the WRONG recovery: it would retrain ARM 1 (another ~40 min of GPU) and
# rewrite the ct_d256_w64_b128_* files, which are already graded and recorded. This file re-runs ARM 2 ALONE and
# then the same chained instruments, so the recovery is one command with no repeated work.
#
# REFUSES TO RUN unless it is the right situation:
#   * a train_chat is already alive -> exit 3 (never two trainers: that OOM-killed scopes here on 2026-10-22);
#   * ARM 1's history is missing    -> exit 4 (the twin comparison would be built on nothing);
#   * ARM 2's best already exists   -> exit 5 (nothing to recover; run the decider, not this).
# Nothing is deleted; the only tag it can write is ct_d256_w64_b256, and only when that arm has no best.
# Predictions: lens/runs/W64-PREDICTIONS-P95.txt (P95a-h) — unchanged, no new predictions are invented here.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

LOG="lens/runs/TIED-W64.txt"
TWIN_BEST="lens/runs/chat/ct_d256_full169_fix3_best.npz"
VRAM="lens/runs/W64-VRAM-SAMPLE.txt"

if [ "$(pgrep -cf '[t]rain_chat')" -gt 0 ]; then
  echo "REFUSING: a train_chat is alive. Never two trainers on this GPU."; exit 3
fi
if [ ! -f lens/runs/chat/ct_d256_w64_b128_history.jsonl ]; then
  echo "REFUSING: ARM 1's history is missing — this is not the ARM-2-recovery situation."; exit 4
fi
if [ -f lens/runs/chat/ct_d256_w64_b256_best.npz ]; then
  echo "REFUSING: ARM 2 already has a best checkpoint — nothing to recover."; exit 5
fi

{
  echo "=============================================================================================="
  echo "=== ARM 2 RECOVERY (ct_d256_w64_b256 ONLY) — $(date '+%F %H:%M:%S')"
  echo "ARM 1 is NOT re-run; its graded files (chat/ct_d256_w64_b128_*) are left untouched."
  echo "ARM 1's recorded best: $(python3 -c "import json;rows=[json.loads(l) for l in open('lens/runs/chat/ct_d256_w64_b128_history.jsonl') if l.strip()];b=max(rows,key=lambda r:r['val_asst_acc']);print(f\"ep{b['epoch']} by-position {b['val_asst_acc']*100:.2f}%\")" 2>/dev/null)"
  echo "=============================================================================================="

  ( for i in $(seq 1 400); do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$VRAM" 2>/dev/null; sleep 45; done ) &
  SAMPLER=$!

  taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 64 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 8450000 --ckpt-seconds 1800 --out ct_d256_w64_b256
  echo "ARM EXIT=$?  tag=ct_d256_w64_b256  batch=256  $(date '+%F %H:%M:%S')"
  kill $SAMPLER 2>/dev/null

  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_w64.py
  echo "grade_w64.py exit=$?"

  if [ ! -f lens/runs/chat/ct_d256_w64_b256_best.npz ]; then
    echo "THE RECOVERY ARM ALSO PRODUCED NO BEST -> the DECIDER is NOT run and ARM 2 is recorded as UNMEASURED."
    echo "=== RECOVERY STOPPED $(date '+%F %H:%M:%S') ===  (nothing substituted; project rule: always the best)"
    exit 7
  fi

  read -r PRIM_TAG PRIM_EP PRIM_ASST PRIM_LEGACY PRIM_LEGACY_FRAC A1_EP A1_ASST A2_EP A2_ASST < <(.venv-gpu/bin/python3 - <<'PY'
import json
def best(tag):
    rows = [json.loads(l) for l in open(f"lens/runs/chat/{tag}_history.jsonl") if l.strip()]
    b = max(rows, key=lambda r: r["val_asst_acc"])
    return b["epoch"], b["val_asst_acc"] * 100, b["val_masked_acc"] * 100
e1, a1, l1 = best("ct_d256_w64_b128")
e2, a2, l2 = best("ct_d256_w64_b256")
if a2 > a1:
    print("ct_d256_w64_b256", e2, f"{a2:.2f}", f"{l2:.2f}", f"{l2/100:.4f}", e1, f"{a1:.2f}", e2, f"{a2:.2f}")
else:
    print("ct_d256_w64_b128", e1, f"{a1:.2f}", f"{l1:.2f}", f"{l1/100:.4f}", e1, f"{a1:.2f}", e2, f"{a2:.2f}")
PY
)
  echo "best by BY-POSITION held-out:  ARM1 b128 ep${A1_EP} ${A1_ASST}%   |   ARM2 b256 ep${A2_EP} ${A2_ASST}%"
  echo "PRIMARY arm for the quantitative instrument: ${PRIM_TAG} ep${PRIM_EP} (by-position ${PRIM_ASST}%, legacy ${PRIM_LEGACY}%)"

  export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-W64.txt"
  export CKPT_LIST="W64 ARM1 b128 best ep${A1_EP} (by-position ${A1_ASST}%)=lens/runs/chat/ct_d256_w64_b128_best.npz;W64 ARM2 b256 best ep${A2_EP} (by-position ${A2_ASST}%)=lens/runs/chat/ct_d256_w64_b256_best.npz;TWIN maskfix d256 window32 best ep3 (by-position 35.26%)=$TWIN_BEST"
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "talk_fair_v3_ship.py exit=$?"

  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${PRIM_TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-W64.txt" \
  TALK_FAIR_LOGGED="$PRIM_LEGACY_FRAC" \
  TALK_FAIR_EP="ep${PRIM_EP} (BEST by-position ${PRIM_ASST}%; legacy by-id ${PRIM_LEGACY}%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/W64-PREDICTIONS-P95.txt (P95d)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py exit=$?"

  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_w64_talk.py
  echo "grade_w64_talk.py exit=$?"

  echo "=== RECOVERY CHAIN DONE $(date '+%F %H:%M:%S') ===  P95g decides whether ANYTHING ships."
} >> "$LOG" 2>&1
