#!/bin/bash
# memfix2_mask_check.sh -- P64b: DID THE MASK MOVE WHEN IT WENT float32 -> bool?
#
# The width fix turns is_asst / is_turnend from float32 (holding exactly 0.0/1.0) into bool, with an explicit
# .astype(np.float32) where the LOSS needs float weights. The mask is the honesty-critical artefact of this
# project (it decides which tokens the model is trained to say), so it is not enough that the code "looks
# equivalent": the two files are run side by side on the SAME real dialogue corpus and their own printed
# numbers are compared.
#
#   A  the PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix_memfix2.py, a byte copy of the shipped file as
#      it was before this turn)                      -> the reference
#   B  the CHANGED script (lens/scripts/train_chat.py)
# Both arms read the same chat_data (lens/scripts, via PYTHONPATH - a copy in _tmp cannot `import chat_data`
# from its own directory), so the ONLY difference between them is the mask dtype.
#
# Corpus: data/chat/prepared_gate_small (215 real hh-rlhf conversations, "Human:"/"Assistant:" markers, so the
# assistant-target count is non-zero and the check CAN fire - a check that cannot fire proves nothing).
# Flags: assistant-only + train-turn-end, CPU, d=128, 2 epochs. Outputs go to their own files under
# lens/runs/_tmp/ and the comparison is appended to lens/runs/MEMFIX2-MASK.txt.
#
# The comparison is made twice: on the whole epoch line, and with the trailing wall-clock column stripped
# (a timing is not a result). The mask lines themselves are compared exactly - they ARE the claim.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small"
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/MEMFIX2-MASK.txt
A_OUT=lens/runs/_tmp/memfix2_mask_A.txt
B_OUT=lens/runs/_tmp/memfix2_mask_B.txt
TINY="--d 128 --window 32 --epochs 2 --batch 256 --lr 0.4 --lam-concept 0.3 --seed 0 --device cpu --assistant-only --train-turn-end"
EPS='^  ep [0-9]+/'
STRIP='s/  ([0-9]*s)$//'
MASK='assistant-only:|train-turn-end:'

{
  echo "=== MEMFIX2 MASK CHECK $(date '+%F %H:%M:%S')  (arms: A=$A_OUT B=$B_OUT)"
  echo "--- A: PRE-CHANGE script (float32 masks)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/runs/_tmp/train_chat_prefix_memfix2.py $TINY --out _memfix2_mask_A > "$A_OUT" 2>&1
  echo "A EXIT=$?"
  echo "--- B: CHANGED script (bool masks)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $TINY --out _memfix2_mask_B > "$B_OUT" 2>&1
  echo "B EXIT=$?"
  echo "--- THE MASK ITSELF (P64b: must be identical)"
  echo "A:"; grep -E "$MASK" "$A_OUT" | sed 's/^/   /'
  echo "B:"; grep -E "$MASK" "$B_OUT" | sed 's/^/   /'
  if diff <(grep -E "$MASK" "$A_OUT") <(grep -E "$MASK" "$B_OUT") > /dev/null; then
    echo "P64b MASK LINES: IDENTICAL"
  else
    echo "P64b MASK LINES: DIFFERENT - see the diff below"
    diff <(grep -E "$MASK" "$A_OUT") <(grep -E "$MASK" "$B_OUT")
  fi
  echo "--- epoch lines, WHOLE (includes the wall clock)"
  if diff <(grep -E "$EPS" "$A_OUT") <(grep -E "$EPS" "$B_OUT"); then echo "A vs B: IDENTICAL"; else echo "A vs B: DIFFERENT"; fi
  echo "--- epoch lines, WALL CLOCK STRIPPED (the numbers)"
  if diff <(grep -E "$EPS" "$A_OUT" | sed "$STRIP") <(grep -E "$EPS" "$B_OUT" | sed "$STRIP"); then
    echo "P64b NUMBERS: IDENTICAL"
  else
    echo "P64b NUMBERS: DIFFERENT"
  fi
  echo "--- can this check fire? assistant targets in arm A (0 would mean it proves nothing)"
  grep -E "$MASK" "$A_OUT" | head -2
  echo "=== DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
tail -30 "$OUT"
