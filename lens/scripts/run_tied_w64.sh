#!/bin/bash
# run_tied_w64.sh — WINDOW 64: THE CONTEXT LEVER, tested at the twin's own scale.  (2026-10-25 self-wake)
#
# WHY THIS ARM EXISTS, from the files, not from taste:
#   * every other lever on the road to talking is CLOSED BY MEASUREMENT: decoder-side (P73/P74,
#     lens/runs/DECODE-ATTRACTOR.txt + DECODE-SEARCH.txt: beam search goes deeper into the hedge, sampling
#     escapes into salad), readout-side (P77, lens/runs/GRADE-P76-P77.txt), attention (P82,
#     lens/runs/GRADE-P82.txt: step-8 2.7% against the twin's 3.9%), and the <unk>-drop objective (P89,
#     lens/runs/GRADE-P89.txt: the <unk> flood fell 2.1% -> 0.0% and the talk STILL does not answer, 0/6).
#   * what is left is MEASURED: the model loses the thread after its own first tokens — free-running step-8
#     2.3-3.9% against teacher-forced 20.7-25.0% (lens/runs/TALK-FAIR-UNKDROP.txt,
#     lens/runs/TALK-FAIR-finalbest-d256-repro.txt) — and the context is only 32 tokens, in which a whole
#     turn (question + the reply being tracked) usually does not fit and the model's own output occupies an
#     ever-larger share as it generates.
#   * THE AXIS HAS NEVER BEEN TRAINED AT SCALE here (32 everywhere) and it is NEXT-EXPERIMENTS.md item H3.
#
# TWO ARMS IN ONE SCRIPT ON PURPOSE — a window change cannot hold tokens, steps and batch equal at once
# (tokens = steps x batch x window), so the confound is measured instead of argued:
#   ARM 1  --batch 128  -> same tokens/epoch (~540M) AND same steps/epoch (~65,600) as the twin
#   ARM 2  --batch 256  -> same tokens/epoch AND same batch as the twin, half the updates (~32,800 steps/epoch)
# Both: --window 64, d=256, 3 epochs, seed 0, lr 0.4, tied direct path, assistant-only, train-turn-end,
# --max-windows 8450000 (at window 64 with stride window//2 the corpus yields about half the windows of the
# window-32 build, so the token budget per epoch matches the twin's 16.9M x 32), --ckpt-seconds 1800 (verified
# P81a-f infra; a shutdown costs at most 30 min and does not change the run), own --out tags.
#
# PREDICTIONS, WRITTEN BEFORE THIS LINE WAS QUEUED: lens/runs/W64-PREDICTIONS-P95.txt (P95a-h).
# WIRING CHECK, NO CORPUS, RUN BEFORE QUEUEING: lens/runs/W64-STRUCTURAL.txt (lens/scripts/w64_structural_check.py):
#   window 32 -> 5,908,686 params (the twin's own log line, reproduced exactly); window 64 -> 8,300,750
#   (+2,392,064 = dmix +2,097,152 + concept.W +294,912), dmix and emb grads finite and non-zero, the
#   window-width guard fires. WIRING ONLY - it says nothing about quality or memory.
#
# ONE CHAINED JOB: both arms, then the pad-free DECIDER on BOTH bests and the stored twin in ONE instrument run,
# so a shutdown between training and measuring cannot leave an arm unmeasured.
# WRITES ONLY new files: lens/runs/TIED-W64.txt, W64-VRAM-SAMPLE.txt, GRADE-P95.txt, PADFREE-TALK-W64.txt,
# TALK-FAIR-W64.txt, and the arms' own chat/ct_d256_w64_b{128,256}_* files. NOTHING IS SHIPPED (P95h).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

LOG="lens/runs/TIED-W64.txt"
TWIN_BEST="lens/runs/chat/ct_d256_full169_fix3_best.npz"
VRAM="lens/runs/W64-VRAM-SAMPLE.txt"

{
  echo "=============================================================================================="
  echo "=== WINDOW 64 x 3 EPOCHS, d=256, FULL CORPUS — $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/W64-PREDICTIONS-P95.txt (P95a-h, written before this was queued)"
  echo "matched control ON DISK (not re-run): $TWIN_BEST   by-position 32.52/34.62/35.26%, talk 4/6 hedges 0/6 answers"
  echo "two arms, one script: ARM 1 batch 128 (same tokens AND steps as the twin), ARM 2 batch 256 (same tokens, half the steps)"
  echo "=============================================================================================="

  : > "$VRAM"
  ( for i in $(seq 1 400); do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$VRAM" 2>/dev/null; sleep 45; done ) &
  SAMPLER=$!

  for ARM in "ct_d256_w64_b128 128" "ct_d256_w64_b256 256"; do
    set -- $ARM
    TAG=$1; BATCH=$2
    echo "----------------------------------------------------------------------------------------------"
    echo "--- ARM tag ${TAG}  batch ${BATCH}  $(date '+%F %H:%M:%S')"
    t0=$(date +%s)
    taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
      --d 256 --window 64 --epochs 3 --batch ${BATCH} --lr 0.4 --lr-decay 0.5 \
      --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
      --direct --direct-mode tied --max-windows 8450000 --ckpt-seconds 1800 --out ${TAG}
    rc=$?
    echo "ARM EXIT=$rc  tag=${TAG}  batch=${BATCH}  wall=$(( $(date +%s) - t0 ))s  $(date '+%F %H:%M:%S')"
  done

  kill $SAMPLER 2>/dev/null
  echo "--- VRAM samples this run (MiB used, util %): $(wc -l < "$VRAM") lines in $VRAM"
  echo "    max used MiB: $(awk -F, 'NR==1{m=$1} {gsub(/ MiB/,"",$1); if ($1+0>m+0) m=$1+0} END{print m}' "$VRAM")"

  # ---- grade the mechanical half (stored text only: no checkpoint, no model, no GPU)
  echo "--- $(date '+%F %H:%M:%S') grade_w64.py (P95a/b/c from the logs and the two arms' own records)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_w64.py
  echo "grade_w64.py exit=$?"

  # ---- the BEST checkpoint of each arm (never the last), and WHICH arm is primary
  if [ ! -f "lens/runs/chat/ct_d256_w64_b128_best.npz" ] || [ ! -f "lens/runs/chat/ct_d256_w64_b256_best.npz" ]; then
    echo "A BEST CHECKPOINT IS MISSING -> the DECIDER cannot run on the artifact that would ship. NOT substituting"
    echo "the last checkpoint (project rule: always the best). Recorded as producing no shippable artifact."
    echo "=== CHAIN STOPPED $(date '+%F %H:%M:%S') ==="
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
  echo "  NOTE the legacy value is passed to the instrument as a FRACTION (${PRIM_LEGACY_FRAC}), not a percent:"

  # ---- THE DECIDER: the pad-free talk test, BOTH bests and the twin in ONE instrument run
  echo "--- $(date '+%F %H:%M:%S') talk_fair_v3_ship.py (THE DECIDER, P95g)"
  export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-W64.txt"
  export CKPT_LIST="W64 ARM1 b128 best ep${A1_EP} (by-position ${A1_ASST}%)=lens/runs/chat/ct_d256_w64_b128_best.npz;W64 ARM2 b256 best ep${A2_EP} (by-position ${A2_ASST}%)=lens/runs/chat/ct_d256_w64_b256_best.npz;TWIN maskfix d256 window32 best ep3 (by-position 35.26%)=$TWIN_BEST"
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "talk_fair_v3_ship.py exit=$?"

  # ---- the step-8 / context numbers that make the decider quantitative (P65c/P65d, P95d)
  echo "--- $(date '+%F %H:%M:%S') talk_fair_test.py on the PRIMARY arm (free-running by step + context ablation)"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/${PRIM_TAG}_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-W64.txt" \
  TALK_FAIR_LOGGED="$PRIM_LEGACY_FRAC" \
  TALK_FAIR_EP="ep${PRIM_EP} (BEST by-position ${PRIM_ASST}%; legacy by-id ${PRIM_LEGACY}%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/W64-PREDICTIONS-P95.txt (P95d)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py exit=$?"

  echo "--- $(date '+%F %H:%M:%S') step-8 and <unk> numbers read from those two files (P95d/e/f)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_w64_talk.py
  echo "grade_w64_talk.py exit=$?"

  echo "=== CHAIN DONE $(date '+%F %H:%M:%S') ===  files: TIED-W64.txt, W64-VRAM-SAMPLE.txt, GRADE-P95.txt,"
  echo "    PADFREE-TALK-W64.txt, TALK-FAIR-W64.txt. P95g decides whether ANYTHING ships."
} >> "$LOG" 2>&1
