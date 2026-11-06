#!/bin/bash
# GRADE THE ATTENTION ARM AND PUT IT THROUGH THE DECIDER — one chained job (2026-10-23 21:0x self-wake).
#
# WHY ONE CHAINED FILE: the arm (lens/scripts/run_attn_full.sh, --out ct_d256_full169_attn) finishes its third
# epoch late tonight and the machine is normally shut down shortly after. Everything below is worthless if it
# has to wait for a later turn, and running it from a systemd scope means it survives this turn's cgroup being
# torn down (measured 2026-10-22: a bare setsid nohup child inherited the cron cgroup and was SIGTERM'd with it).
#
# IT DOES, IN ORDER, AND NOTHING ELSE:
#   0. waits for the trainer process to exit (polling pgrep for that exact --out tag), sampling nvidia-smi every
#      15 s meanwhile -> lens/runs/ATTN-VRAM-SAMPLE.txt (needed by P82b, which has never had a peak measurement).
#   1. grade_attn_full.py            -> lens/runs/GRADE-P82.txt  (P82a-d, P88a-b, from stored text only)
#   2. talk_fair_test.py             -> lens/runs/TALK-FAIR-ATTN-FULL.txt (P65c/D/E + the step-8 number P82e
#                                       is decided by), on the arm's BEST checkpoint, with the arm's own legacy
#                                       column passed as the reproduction control.
#   3. talk_fair_v3_ship.py          -> lens/runs/PADFREE-TALK-ATTN.txt (pad-free greedy samples for BOTH the arm
#                                       and the twin, in the SAME run, so the two <unk> shares are comparable)
#   4. grade_attn_talk.py            -> appends the P82e/P82f/P88c-g section to GRADE-P82.txt
#
# PREDICTIONS, WRITTEN BEFORE THIS FILE RAN: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (P82a-f, the arm's own) and
# lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt (P88a-g, the instruments'). Neither is edited by this file.
# NOTHING IS SHIPPED (P82f): no export, no model card, no server. No stored run, log, checkpoint or number is
# overwritten: every output above is a NEW file (GRADE-P82.txt did not exist), and the instruments are CPU-only,
# nice 19, pinned to cores 4,5 so a trainer on the GPU is not disturbed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2

ATTN_TAG="ct_d256_full169_attn"
BEST="lens/runs/chat/${ATTN_TAG}_best.npz"
TWIN_BEST="lens/runs/chat/ct_d256_full169_fix3_best.npz"
HIST="lens/runs/chat/${ATTN_TAG}_history.jsonl"

{
  echo "=============================================================================================="
  echo "=== ATTENTION ARM: GRADE + DECIDER CHAIN  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (arm, P82a-f) +"
  echo "             lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt (instruments, P88a-g)"
  echo "twin comparator, stored, measured in the SAME run at step 3: $TWIN_BEST"
  echo "=============================================================================================="

  # ---- 0. wait for the trainer to exit, sampling GPU memory for P82b
  peak=0
  waited=0
  while pgrep -f "train_chat\.py .*--out ${ATTN_TAG}\$" >/dev/null 2>&1; do
    if [ $(( waited % 15 )) -eq 0 ]; then
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      if [ -n "$m" ]; then
              if [ "$m" -gt "$peak" ]; then peak=$m; fi
            fi
    fi
    sleep 5
    waited=$(( waited + 5 ))
    if [ "$waited" -ge 5400 ]; then
      echo "$(date '+%F %H:%M:%S') WAITER TIMEOUT after 90 min: the trainer is still alive; NOT running the"
      echo "  instruments on a half-written checkpoint. Nothing written below. Re-run this file by hand later."
      exit 3
    fi
  done
  echo "$(date '+%F %H:%M:%S') trainer with --out ${ATTN_TAG} has exited (waited ${waited}s)."
  echo "peak GPU memory sampled while it trained: ${peak} MiB"
  echo "${peak} MiB peak, sampled every 15 s by run_attn_grade_talk.sh (GPU total 6144 MiB)" \
    > lens/runs/ATTN-VRAM-SAMPLE.txt
  sleep 5

  # ---- 1. grade P82a-d from the stored text
  echo "--- $(date '+%F %H:%M:%S') step 1: grade_attn_full.py"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_attn_full.py
  echo "grade_attn_full.py exit=$?"

  # ---- the best (not the last) epoch, and its LEGACY column for the reproduction control
  read -r BEST_EP BEST_ASST BEST_LEGACY < <(.venv-gpu/bin/python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("lens/runs/chat/ct_d256_full169_attn_history.jsonl") if l.strip()]
b = max(rows, key=lambda r: r["val_asst_acc"])
print(b["epoch"], f'{b["val_asst_acc"]*100:.2f}', f'{b["val_masked_acc"]*100:.2f}')
PY
)
  echo "best epoch by BY-POSITION held-out: ep${BEST_EP} (${BEST_ASST}%; its LEGACY by-token-id column ${BEST_LEGACY}%)"
  echo "checkpoint being tested: $BEST"

  # ---- 2. talk_fair_test.py: the free-running-by-step number P82e is decided by
  echo "--- $(date '+%F %H:%M:%S') step 2: talk_fair_test.py (P65c/d/e + step-8)"
  TALK_FAIR_CKPT="$PWD/$BEST" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-ATTN-FULL.txt" \
  TALK_FAIR_LOGGED="$BEST_LEGACY" \
  TALK_FAIR_EP="ep${BEST_EP} (BEST by-position ${BEST_ASST}%; legacy by-id ${BEST_LEGACY}%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (P82e) + lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt (P88c)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py exit=$?"

  # ---- 3. the pad-free ship instrument on BOTH checkpoints in one run
  echo "--- $(date '+%F %H:%M:%S') step 3: talk_fair_v3_ship.py (pad-free, arm vs twin)"
  export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-ATTN.txt"
  export CKPT_LIST="ATTENTION best (--attention on, held-out by-position ${BEST_ASST}%)=$BEST;TWIN maskfix full-corpus best (plain SGD, no attention, held-out by-position 35.26%)=$TWIN_BEST"
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "talk_fair_v3_ship.py exit=$?"

  # ---- 4. the verdicts for P82e/f and P88c-g, from those two texts
  echo "--- $(date '+%F %H:%M:%S') step 4: grade_attn_talk.py"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_attn_talk.py
  echo "grade_attn_talk.py exit=$?"

  echo "=== CHAIN DONE $(date '+%F %H:%M:%S') ===  files: GRADE-P82.txt, TALK-FAIR-ATTN-FULL.txt,"
  echo "    PADFREE-TALK-ATTN.txt, ATTN-VRAM-SAMPLE.txt. P82f: NOTHING SHIPPED."
} >> lens/runs/ATTN-GRADE-RUN.txt 2>&1
