#!/bin/bash
# run_w64_talk_repair.sh — RE-RUN THE WINDOW-64 DECIDER after the 11:45:56 instrument crash.  (2026-10-25 self-wake)
#
# WHY: lens/scripts/run_tied_w64.sh finished BOTH arms (ARM EXIT=0) and graded P95a/b/c, but its three talk
# instruments all died, so P95d (the point of the experiment), P95e, P95f and P95g (the decider that decides
# whether anything ships) are UNMEASURED. The three failures, with their tracebacks in lens/runs/TIED-W64.txt:
#   talk_fair_v3_ship.py  exit=1  "ValueError: windows must be 64 tokens wide, got 32" (hardcoded W=32)
#   talk_fair_test.py     exit=1  same ValueError (hardcoded W=32)
#   grade_w64_talk.py     exit=1  "SyntaxError: f-string expression part cannot include a backslash" (py3.11)
# The repair reads the window PER CHECKPOINT from its own metadata and builds the grader's verdict outside the
# f-string. PREDICTIONS, WRITTEN BEFORE ANY EDIT: lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt (P96a-g).
#
# ORDER MATTERS: the CONTROL runs FIRST, on the stored W=32 twin, and it is falsifiable to the digit (P96a). If
# the control does not reproduce the stored numbers, the window-64 numbers are not reported at all.
#
# WRITES ONLY NEW FILES: PADFREE-TALK-W64.txt, PADFREE-TALK-W64-CONTROL.txt, TALK-FAIR-W64.txt,
# TALK-FAIR-W64-CONTROL.txt, GRADE-P95-TALK.txt and this log. NOTHING stored is overwritten, no checkpoint is
# written, no training is started, and the GPU is untouched (the gate-cost ablation holds it).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

LOG="lens/runs/W64-TALK-REPAIR.txt"
TWIN_BEST="lens/runs/chat/ct_d256_full169_fix3_best.npz"
A1="lens/runs/chat/ct_d256_w64_b128_best.npz"
A2="lens/runs/chat/ct_d256_w64_b256_best.npz"
PY=".venv-gpu/bin/python3 -u"

{
  echo "=============================================================================================="
  echo "=== WINDOW-64 TALK DECIDER, REPAIRED AND RE-RUN — $(date '+%F %H:%M:%S')"
  echo "predictions written BEFORE any edit: lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt (P96a-g)"
  echo "the runner that produced the arms: lens/scripts/run_tied_w64.sh (log lens/runs/TIED-W64.txt)"
  echo "=============================================================================================="

  echo ""
  echo "################ CONTROL 1/2 (P96a i): the repaired v3 on the STORED W=32 TWIN — must reproduce"
  echo "################ PADFREE-TALK-ATTN.txt lines 131-137 (mean <unk> 2.1%, 6/6 >=4-word runs)"
  echo "--- $(date '+%F %H:%M:%S') talk_fair_v3_ship.py, CKPT_LIST = the twin alone"
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-W64-CONTROL.txt" \
  CKPT_LIST="TWIN maskfix d256 window32 best ep3 (by-position 35.26%)=$TWIN_BEST" \
  nice -n 19 ionice -c3 taskset -c 4,5 $PY lens/scripts/talk_fair_v3_ship.py
  echo "control v3 exit=$?"

  echo ""
  echo "################ CONTROL 2/2 (P96a ii): the repaired talk_fair_test.py on the SAME twin — must"
  echo "################ reproduce TALK-FAIR-finalbest-d256-repro.txt (real 35.26 / mismatched 2.20 / drop"
  echo "################ 33.07 / step-8 free-running 3.9% against teacher-forced 23.0%)"
  echo "--- $(date '+%F %H:%M:%S') talk_fair_test.py on the twin"
  TALK_FAIR_CKPT="$PWD/$TWIN_BEST" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-W64-CONTROL.txt" \
  TALK_FAIR_LOGGED="0.2631" \
  TALK_FAIR_EP="ep3 (FINAL BEST; legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt (P96a ii control)" \
    nice -n 19 ionice -c3 taskset -c 4,5 $PY lens/scripts/talk_fair_test.py
  echo "control talk_fair_test exit=$?"

  echo ""
  echo "################ THE REAL RUNS: the two WINDOW-64 arms, each measured at ITS OWN window"
  echo "--- $(date '+%F %H:%M:%S') talk_fair_v3_ship.py, CKPT_LIST = ARM1 + ARM2 + the twin (THE DECIDER, P95g)"
  PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-W64.txt" \
  CKPT_LIST="W64 ARM1 b128 best ep3 (by-position 32.27%)=$A1;W64 ARM2 b256 best ep3 (by-position 31.38%)=$A2;TWIN maskfix d256 window32 best ep3 (by-position 35.26%)=$TWIN_BEST" \
  nice -n 19 ionice -c3 taskset -c 4,5 $PY lens/scripts/talk_fair_v3_ship.py
  echo "w64 v3 exit=$?"

  echo ""
  echo "--- $(date '+%F %H:%M:%S') talk_fair_test.py on the PRIMARY arm (ARM1, the best by-position) (P95d)"
  TALK_FAIR_CKPT="$PWD/$A1" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-W64.txt" \
  TALK_FAIR_LOGGED="0.2293" \
  TALK_FAIR_EP="ep3 (BEST by-position 32.27%; legacy by-id 22.93%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/W64-PREDICTIONS-P95.txt (P95d)" \
    nice -n 19 ionice -c3 taskset -c 4,5 $PY lens/scripts/talk_fair_test.py
  echo "w64 talk_fair_test exit=$?"

  echo ""
  echo "--- $(date '+%F %H:%M:%S') grade_w64_talk.py (P95d/e/f/g, stored text only) — the file that could not run"
  nice -n 19 ionice -c3 $PY lens/scripts/grade_w64_talk.py
  echo "grade_w64_talk.py exit=$?"

  echo ""
  echo "=== REPAIR CHAIN DONE $(date '+%F %H:%M:%S') ===  files: W64-TALK-REPAIR.txt,"
  echo "    PADFREE-TALK-W64.txt, PADFREE-TALK-W64-CONTROL.txt, TALK-FAIR-W64.txt,"
  echo "    TALK-FAIR-W64-CONTROL.txt, GRADE-P95-TALK.txt"
} >> "$LOG" 2>&1
