#!/bin/bash
# run_grade_w64_talk_rerun.sh — re-run the W64 talk grader after fixing the sample-line regex.
# WHY: the first run of grade_w64_talk.py dropped every sample whose longest non-<unk> run printed as a
# single digit ("run  6" vs the pattern's "run 6"), so ARM1 showed 4 samples and ARM2 3 where both arms have
# 6, and the hedge denominators were 4 and 3 instead of 6. The mean <unk> numbers were never affected.
# [BUG-GRADER-DROPS-SINGLE-DIGIT-RUNS]  Text only: no checkpoint, no model, no GPU. Appends to
# lens/runs/GRADE-P95-TALK.txt; the LAST block there is the authoritative one.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -m py_compile lens/scripts/grade_w64_talk.py || exit 2
echo "py_compile OK"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/grade_w64_talk.py
echo "grade_w64_talk.py exit=$?"
