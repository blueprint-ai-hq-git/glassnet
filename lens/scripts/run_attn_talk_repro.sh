#!/bin/bash
# CORRECTION PASS on the attention arm's grading (2026-10-23 21:0x self-wake, immediately after the first pass).
#
# TWO DEFECTS FOUND IN THE FIRST PASS, BOTH MINE, BOTH FIXED HERE:
#  (1) MY INPUT ERROR: the first run of lens/scripts/talk_fair_test.py passed TALK_FAIR_LOGGED="24.31" (a
#      PERCENTAGE). The instrument reads that env value as a FRACTION (talk_fair_test.py line 39) and compares
#      abs(macc_legacy - LOGGED_MASKED) <= 0.002, so it printed "the trainer logged 2431.00%" and then its own
#      guard line: "MISMATCH : MY HARNESS DIFFERS FROM THE TRAINER'S - nothing below may be reported". The
#      stored file lens/runs/TALK-FAIR-ATTN-FULL.txt is KEPT, untouched, as the record of that failure (the
#      project has done exactly this before: lens/runs/TALK-FAIR-finalbest-d256.txt kept, repro in
#      TALK-FAIR-finalbest-d256-repro.txt). CORRECTED value: 0.2431.
#  (2) RACE IN MY WAITER: it stops as soon as the PYTHON trainer process is gone, and the arm's bash wrapper
#      writes its "=== DONE" line a moment later, so the first grade run found no DONE line and reported P82b as
#      one side unmeasured. grade_attn_full.py now falls back to the sum of the per-epoch seconds, labelled.
#
# IT RE-RUNS, IN ORDER: the corrected talk_fair_test (NEW file, first output untouched), then the two graders
# (grade_attn_full.py WRITES GRADE-P82.txt, grade_attn_talk.py APPENDS to it), so GRADE-P82.txt ends as ONE
# coherent document graded against the corrected control. CPU only, nice 19, cores 4,5: the gate-cost ablation
# the watchdog started at 21:01:44 is training on the GPU and is NOT touched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2

{
  echo "--- $(date '+%F %H:%M:%S') CORRECTION PASS: talk_fair_test.py with TALK_FAIR_LOGGED=0.2431 (a fraction)"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_attn_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-ATTN-FULL-REPRO.txt" \
  TALK_FAIR_LOGGED="0.2431" \
  TALK_FAIR_EP="ep3 (BEST by-position 33.00%; legacy by-id 24.31%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (P82e) + lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt (P88c)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py (repro) exit=$?"
  grep -m1 "P65e reproduction" lens/runs/TALK-FAIR-ATTN-FULL-REPRO.txt

  echo "--- $(date '+%F %H:%M:%S') re-grade P82a-d (writes GRADE-P82.txt)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_attn_full.py
  echo "grade_attn_full.py exit=$?"

  echo "--- $(date '+%F %H:%M:%S') re-grade P82e/f + P88c-g against the CORRECTED file (appends)"
  FAIR_TALK_TXT="$PWD/lens/runs/TALK-FAIR-ATTN-FULL-REPRO.txt" \
  PAD_TALK_TXT="$PWD/lens/runs/PADFREE-TALK-ATTN.txt" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_attn_talk.py
  echo "grade_attn_talk.py exit=$?"
  echo "=== CORRECTION PASS DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/ATTN-GRADE-RUN.txt 2>&1
