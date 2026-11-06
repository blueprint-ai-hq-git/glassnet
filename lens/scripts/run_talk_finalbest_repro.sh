#!/bin/bash
# THE SAME TALK TEST, ONE CORRECTED INPUT (P84e follow-up, 2026-10-23 17:3x self-wake).
#
# WHY THIS FILE EXISTS: the first run of this test (lens/runs/TALK-FAIR-finalbest-d256.txt) printed
# "P65e reproduction: FAIL (LEGACY by-token-id column 26.31% against the logged 35.26%)". That FAIL is MY
# harness input error, not a finding about the model: the instrument compares its LEGACY by-token-id column
# against TALK_FAIR_LOGGED, and I passed 35.26% -- which is the BY-POSITION number from the trainer's epoch
# line (lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt ep 3/3:
#   MASKED(legacy by-id) ppl 84.80 acc 26.31% (n=44418)   ASST(by-position) ppl 32.77 acc 35.26% (n=58191)).
# So the instrument was made to compare 26.31% with the other column's number. This is the verdict-check
# failure mode this project keeps hitting: a right number given the wrong comparison. The correction is the
# legacy column's own value, 0.2631, and the check must then pass with both numbers shown.
#
# It writes a NEW file (tag "repro"); the first output is kept untouched. CPU only, nice 19 / ionice -c3.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TALK TEST, FINAL BEST, CORRECTED REPRO INPUT $(date '+%F %H:%M') ==="
  echo "TALK_FAIR_LOGGED=0.2631 = the trainer's LEGACY by-token-id column at ep3 (the column this instrument"
  echo "reports as 'BY TOKEN ID (LEGACY)'); the earlier run passed 0.3526, the BY-POSITION number, by mistake."
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-finalbest-d256-repro.txt" \
  TALK_FAIR_LOGGED=0.2631 TALK_FAIR_EP="ep3 (FINAL BEST; legacy by-id 26.31%, by-position 35.26%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/TALK-FINALBEST-PREDICTIONS-P84.txt" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TALK-FINALBEST-RUN.txt 2>&1
