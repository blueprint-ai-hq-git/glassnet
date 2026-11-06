#!/bin/bash
# selfctx_preflight.sh — CPU WIRING PRE-FLIGHT for the new --self-context-* flags (exposure bias).
# Predictions P98a-c in lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt, written BEFORE this file existed.
# THIS IS A WIRING TEST, NOT A QUALITY CLAIM: d=64 / 40,000 windows / 1 epoch. Nothing here is reported as a
# result about the model, and no number from here goes in STATE.md as a quality number.
# FIVE tiny runs, sequential (memory: each loads the whole corpus cache, so they must not overlap):
#   1. PRE-CHANGE copy of the trainer, flag absent  -> the reference epoch line
#   2. new trainer, flag absent                     -> P98a: must be digit-identical to (1)
#   3. new trainer, --self-context-frac 1.0 --self-context-rounds 2 -> P98b: real, and 0 < agreement < 100%
#   4. new trainer, --self-context-frac 0.5 --mtp 2 -> P98c: REFUSED (mtp), no step, cheap
#   5. new trainer, --self-context-frac 0.5 --self-context-rounds 99 -> P98c: REFUSED (rounds)
# CPU ONLY, cores 2,3, nice 19. The GPU holds the ours-method drip and this must not disturb it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
# the PRE-CHANGE copy lives in lens/runs/_tmp/, from where `import chat_data` / `chat_concepts` are NOT
# importable (the real trainer finds them because python prepends the script's own directory) - the fix is
# measured and documented in lens/scripts/unk_drop_preflight.sh; for lens/scripts/train_chat.py this is a no-op.
export PYTHONPATH="$PWD/lens/scripts"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"

OUT=lens/runs/SELF-CONTEXT-PREFLIGHT.txt
TMP=lens/runs/_tmp
PRE="$TMP/train_chat_prefix_selfctx_20261025-1300.py"
[ -f "$PRE" ] || { echo "PRE-CHANGE COPY MISSING: $PRE - refusing to run (P98a is unanswerable without it)"; exit 2; }

COMMON="--d 64 --window 32 --epochs 1 --batch 128 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
--seed 0 --device cpu --assistant-only --train-turn-end --direct --direct-mode tied \
--max-windows 40000 --method standard --ckpt-seconds 0"

{
echo "================================================================================================"
echo "SELF-CONTEXT PRE-FLIGHT (P98a-c)  $(date '+%F %H:%M:%S')   d=64, 40,000 windows, 1 epoch, CPU, cores 2,3"
echo "predictions: lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt   (written before this runner existed)"
echo "WIRING ONLY: d=64 quality is meaningless and is not reported as anything else"
echo "================================================================================================"
} >> "$OUT"

run () {   # $1 = tag, $2 = python file, $3 = extra flags
  local tag="$1" py="$2" extra="$3"
  local f="$TMP/sctxpf_${tag}.out"
  # FRESH FILE PER LAUNCH (2026-10-25, fixed after the first two blocks): this used to append, so the second
  # block's grader still found the FIRST block's TypeError traceback in the file and read a clean run as a
  # crash. The authoritative record is the appended main log; these per-run files are scratch for the grader,
  # and the earlier failure stays on the record in the main log and in BUGS.md.
  : > "$f"
  {
  echo "------------------------------------------------------------------------------------------------"
  echo "ARM $tag   interpreter=$py   extra='${extra:-<none>}'"
  echo "------------------------------------------------------------------------------------------------"
  } >> "$f"
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u "$py" $COMMON $extra --out "sctxpf_${tag}" >> "$f" 2>&1
  echo "exit=$?" >> "$f"
  tail -25 "$f" >> "$OUT"
}

run prechange     "$PRE"                          ""
run newdefault    lens/scripts/train_chat.py      ""
run selfctx_on    lens/scripts/train_chat.py      "--self-context-frac 1.0 --self-context-rounds 2"
run guard_mtp     lens/scripts/train_chat.py      "--self-context-frac 0.5 --mtp 2"
run guard_rounds  lens/scripts/train_chat.py      "--self-context-frac 0.5 --self-context-rounds 99"

echo "" >> "$OUT"
echo "=== GRADING (P98a-c) ===" >> "$OUT"
.venv-gpu/bin/python3 "$PWD/lens/scripts/selfctx_preflight_grade.py" >> "$OUT" 2>&1
echo "=== DONE $(date '+%F %H:%M:%S') ===" >> "$OUT"
