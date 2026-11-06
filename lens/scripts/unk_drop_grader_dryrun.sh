#!/bin/bash
# unk_drop_grader_dryrun.sh — prove grade_unk_drop_talk.py's PARSERS work, BEFORE the --drop-unk arm produces
# anything, by pointing it at a STORED pair it did not write: the attention arm's pad-free files. It must
# reproduce the numbers those files already state (attention mean <unk> 6.9%, twin 2.1%, attention conv 10955
# SPIRAL 33.3%/run 3, twin conv 10955 0.0%/run 24), and the verdicts it prints must be the ones ALREADY on
# record in lens/runs/GRADE-P82.txt (P88d HELD, P88e FAILED on the mechanical half).
# It writes ONLY lens/runs/_tmp/grader_dryrun_unkdrop_talk.txt. No model, no GPU, no checkpoint.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export UNKDROP_PADFREE="$PWD/lens/runs/PADFREE-TALK-ATTN.txt"
export UNKDROP_FAIR="$PWD/lens/runs/TALK-FAIR-ATTN-FULL.txt"
export UNKDROP_GRADE_OUT="$PWD/lens/runs/_tmp/grader_dryrun_unkdrop_talk.txt"
: > "$UNKDROP_GRADE_OUT"
{
  echo "=== DRY RUN of grade_unk_drop_talk.py against the ATTENTION arm's stored files $(date '+%F %H:%M:%S') ==="
  echo "these are NOT the drop-unk arm's numbers; this only proves the parsers and the comparison table."
  echo "NOTE: the grader prints to stdout AND appends to its own OUT, so stdout goes to a SEPARATE run file"
  echo "      here (the first version pointed both at the same file and every line appeared twice; the"
  echo "      numbers were unaffected - recorded so the fix is visible)."
  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/grade_unk_drop_talk.py
  echo "grader exit=$?"
} >> "${UNKDROP_GRADE_OUT%.txt}-RUN.txt" 2>&1

# ---- SECOND DRY RUN: grade_unk_drop.py (P89d/P89e) against the PRE-FLIGHT's own tiny d=64 arm --------------
# The point is to prove the epoch-line regex, the drop-unk line parser and the twin comparison run on REAL
# log lines BEFORE the full-corpus arm exists. It is NOT a verdict: the pre-flight log holds three arms and the
# numbers are d=64 wiring numbers, so the "CONTROL" line is expected to read FAIL/INCOMPLETE here and that is
# the honest outcome of a parser test, not a finding.
export UNKDROP_LOG="$PWD/lens/runs/UNK-DROP-PREFLIGHT.txt"
export UNKDROP_ARM_HIST="$PWD/lens/runs/chat/unkpf_dropunk_history.jsonl"
export UNKDROP_GRADE_OUT="$PWD/lens/runs/_tmp/grader_dryrun_unkdrop.txt"
: > "$UNKDROP_GRADE_OUT"
{
  echo "=== DRY RUN 2 of grade_unk_drop.py against the PRE-FLIGHT's d=64 dropunk arm $(date '+%F %H:%M:%S') ==="
  echo "parser test only: the numbers below are d=64 wiring numbers and are NOT reported as results anywhere."
  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/grade_unk_drop.py
  echo "grader exit=$?"
} >> "${UNKDROP_GRADE_OUT%.txt}-RUN2.txt" 2>&1
