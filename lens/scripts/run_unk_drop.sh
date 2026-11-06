#!/bin/bash
# run_unk_drop.sh — CANDIDATE (ii) OF P78f/P82e: STOP PAYING FOR THE <unk> TARGETS.  (2026-10-24 self-wake)
#
# WHY THIS ARM EXISTS, from the files, not from taste:
#   * lens/runs/GRADE-P82.txt: the attention arm is graded and its lever is CLOSED — free-running step-8 2.7%
#     against the twin's 3.9% (P82e's own falsifier fired), ep3 by-position 33.00% against 35.26%, mean <unk>
#     6.9% against 2.1%, 0/6 hedges, 0/6 topical answers. P82e's pre-written next candidate is (ii).
#   * lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (P82e) and lens/runs/SHIP-TALK-v3.txt (P71f): 6.888% of held-out
#     assistant targets are id 0 (<unk>, an out-of-vocabulary word) and answer.b[<unk>] = +6.1075, rank 0 of
#     12,002 — the single largest bias in the model.
#   * lens/scripts/run_code_dropunk.sh: the same lever measured once already on the code corpus (<unk> was the
#     #1 most frequent training target at 6.98%), with the same conclusion: those positions are unanswerable by
#     construction, so the plain objective can only teach the model to EMIT <unk>.
#
# WHAT CHANGES, AND WHAT DOES NOT: ONE flag, --drop-unk (new in lens/scripts/train_chat.py this turn, default
# OFF, wiring pre-flighted in lens/runs/UNK-DROP-PREFLIGHT.txt). Validation is NOT changed by it, so every
# held-out column stays comparable. Everything else is the twin's recipe flag for flag:
#   --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0
#   --seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 16900000
# PLUS --ckpt-seconds 1800 (mid-epoch checkpoints, the P81a-f verified infra: a shutdown costs at most 30 min,
# it does NOT change the run) and its own --out tag. The twin, ct_d256_full169_fix3, is the MATCHED CONTROL and
# is already on disk: ep3 train 38.57%, held-out BY POSITION 35.26%, legacy by-id 26.31%
# (lens/runs/chat/ct_d256_full169_fix3_history.jsonl); talk 4/6 hedges 0/6 answers, mean <unk> 2.1%
# (lens/runs/GRADE-P76-P77.txt); free-running step-8 3.9% (lens/runs/TALK-FAIR-finalbest-d256-repro.txt).
#
# PREDICTIONS, WRITTEN BEFORE THIS LINE WAS QUEUED: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89a-h).
# IT IS ONE CHAINED JOB ON PURPOSE: the arm, then its grading, then the DECIDER, so a shutdown between them
# cannot leave the arm trained with nothing measuring it (the attention arm's chain, run_attn_grade_talk.sh).
# WRITES ONLY: lens/runs/UNK-DROP.txt, lens/runs/GRADE-P89.txt, lens/runs/PADFREE-TALK-UNKDROP.txt (+ its
# .json), lens/runs/TALK-FAIR-UNKDROP.txt, and the arm's own chat/unk_drop_169_3_* files. NOTHING SHIPPED.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

TAG=unk_drop_169_3
HIST="lens/runs/chat/${TAG}_history.jsonl"
BEST="lens/runs/chat/${TAG}_best.npz"
TWIN_BEST="lens/runs/chat/ct_d256_full169_fix3_best.npz"
LOG="lens/runs/UNK-DROP.txt"

{
  echo "=============================================================================================="
  echo "=== CANDIDATE (ii): STOP PAYING FOR THE <unk> TARGETS — $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89a-h, written before this was queued)"
  echo "matched control ON DISK (not re-run): $TWIN_BEST"
  echo "one flag differs: --drop-unk. Plus --ckpt-seconds 1800 (checkpoint safety only)."
  echo "=============================================================================================="

  t0=$(date +%s)
  taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --ckpt-seconds 1800 --drop-unk \
    --out ${TAG}
  rc=$?
  echo "ARM EXIT=$rc   wall=$(( $(date +%s) - t0 ))s   $(date '+%F %H:%M:%S')"

  # ---- 1. GRADE from the stored text only (no checkpoint opened, no model, no GPU)
  echo "--- $(date '+%F %H:%M:%S') step 1: grade_unk_drop.py (P89d/P89e + its own control)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_unk_drop.py
  echo "grade_unk_drop.py exit=$?"

  # ---- 2. THE BEST epoch (never the last) and its LEGACY column, for the reproduction control
  if [ ! -f "$BEST" ]; then
    echo "NO BEST CHECKPOINT: $BEST does not exist, so the DECIDER cannot run on the artifact that would"
    echo "ship. NOT substituting the last checkpoint (project rule: always the best). Stop here; this run"
    echo "produced no shippable artifact and that is recorded as such."
    echo "=== CHAIN STOPPED $(date '+%F %H:%M:%S') ==="
    exit 7
  fi
  read -r BEST_EP BEST_ASST BEST_LEGACY BEST_LEGACY_FRAC < <(.venv-gpu/bin/python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("lens/runs/chat/unk_drop_169_3_history.jsonl") if l.strip()]
b = max(rows, key=lambda r: r["val_asst_acc"])
print(b["epoch"], f'{b["val_asst_acc"]*100:.2f}', f'{b["val_masked_acc"]*100:.2f}', f'{b["val_masked_acc"]:.4f}')
PY
)
  echo "best epoch by BY-POSITION held-out: ep${BEST_EP} (${BEST_ASST}%; its LEGACY by-token-id column ${BEST_LEGACY}%)"
  echo "  (the legacy value is passed to the instrument as a FRACTION, ${BEST_LEGACY_FRAC}: BUG-TALKFAIR-LOGGED-UNITS)"

  # ---- 3. THE DECIDER: pad-free talk, THE ARM AND THE TWIN IN THE SAME RUN, one instrument
  echo "--- $(date '+%F %H:%M:%S') step 3: talk_fair_v3_ship.py (THE DECIDER, P89f)"
  export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-UNKDROP.txt"
  export CKPT_LIST="UNK-DROP arm (--drop-unk, held-out by-position ${BEST_ASST}%)=$BEST;TWIN maskfix full-corpus best (no --drop-unk, held-out by-position 35.26%)=$TWIN_BEST"
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "talk_fair_v3_ship.py exit=$?"

  # ---- 4. the step-8 / context numbers that make the decider quantitative (P65c/P65d)
  echo "--- $(date '+%F %H:%M:%S') step 4: talk_fair_test.py (free-running by step + context ablation)"
  TALK_FAIR_CKPT="$PWD/$BEST" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-UNKDROP.txt" \
  TALK_FAIR_LOGGED="$BEST_LEGACY_FRAC" \
  TALK_FAIR_EP="ep${BEST_EP} (BEST by-position ${BEST_ASST}%; legacy by-id ${BEST_LEGACY}%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89f)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py exit=$?"

  # ---- 5. the verdicts that CAN be read mechanically, printed beside the arm and the twin
  echo "--- $(date '+%F %H:%M:%S') step 5: grade_unk_drop_talk.py (P89f/g, from the two stored texts)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_unk_drop_talk.py
  echo "grade_unk_drop_talk.py exit=$?"

  echo "=== CHAIN DONE $(date '+%F %H:%M:%S') ===  files: UNK-DROP.txt, GRADE-P89.txt,"
  echo "    PADFREE-TALK-UNKDROP.txt, TALK-FAIR-UNKDROP.txt. P89g: NOTHING IS SHIPPED from this arm."
} >> "$LOG" 2>&1
