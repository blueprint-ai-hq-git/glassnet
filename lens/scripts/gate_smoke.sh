#!/bin/bash
# gate_smoke.sh -- the smoke test for PLAN item (d2b) part 1: WIRE THE MEMORISATION RULE INTO THE SHIPPED
# TRAINER. Predictions first: lens/runs/TRAIN-GATES-PREDICTIONS.txt (P58a, P58c).
#
# THREE invocations on the SMALL corpus (data/chat/prepared_gate_small, ~1,200 train / ~360 val windows of 32):
#   A  the PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix_gates.py, md5 b25806a7926981bbcf956a457decdeca
#      = the shipped file exactly as it was before this turn), default path                  -> the reference
#   B  the CHANGED script, NO new flags                                                    -> P58a: A vs B
#   C  the CHANGED script, --gates gap --gate-variant off (the CONTROL)                    -> P58c: A vs C
#
# EACH ARM'S OUTPUT GOES TO ITS OWN FILE and is then appended to lens/runs/GATE-SMOKE.txt, so the comparison
# reads the three files directly and a re-run of this script cannot corrupt the comparison by adding a second
# block to the shared log (the first version compared by line RANGE inside the shared log, which a re-run
# would have broken).
# The comparison is made TWICE on purpose: once on the whole epoch line, and once with the trailing wall-clock
# column stripped. The wall-clock column is a timing, not a result, and the gate arm is slower by construction
# - a bar that fails on it would be failing on the clock, not on the numbers. Both verdicts are printed.
#
# WHY PYTHONPATH: the pre-change copy lives in lens/runs/_tmp/, so its own directory is not lens/scripts and
# `import chat_data` fails there (measured: ModuleNotFoundError). Putting lens/scripts on PYTHONPATH resolves
# the same modules the changed script finds through its own directory. No file is modified for this.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small"
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/GATE-SMOKE.txt
A_OUT=lens/runs/_tmp/gate_smoke_A.txt
B_OUT=lens/runs/_tmp/gate_smoke_B.txt
C_OUT=lens/runs/_tmp/gate_smoke_C.txt
TINY="--d 128 --window 32 --epochs 3 --batch 256 --lr 0.4 --lam-concept 0.3 --seed 0 --device cpu"
EPS='^  ep [0-9]+/'
STRIP='s/  ([0-9]*s)$//'

{
  echo "=== GATE SMOKE START $(date '+%F %H:%M:%S')  (per-arm output in lens/runs/_tmp/gate_smoke_{A,B,C}.txt)"
  echo "--- A: PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix_gates.py), default path"
  nice -n 19 ionice -c3 python3 -u lens/runs/_tmp/train_chat_prefix_gates.py $TINY --out _gate_smoke_A > "$A_OUT" 2>&1
  echo "A EXIT=$?"
  echo "--- B: CHANGED script, NO new flags (must be identical to A: P58a)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --out _gate_smoke_B > "$B_OUT" 2>&1
  echo "B EXIT=$?"
  echo "--- C: CHANGED script, --gates gap --gate-variant off (CONTROL, must refuse 0 steps: P58c)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates gap --gate-variant off \
    --probe-windows 256 --out _gate_smoke_C > "$C_OUT" 2>&1
  echo "C EXIT=$?"
  echo "--- A vs B, WHOLE EPOCH LINE (P58a)"
  if diff <(grep -E "$EPS" "$A_OUT") <(grep -E "$EPS" "$B_OUT"); then echo "A vs B: IDENTICAL"; else echo "A vs B: DIFFERENT (see the diff above)"; fi
  echo "--- A vs C, WHOLE EPOCH LINE (P58c, includes the wall-clock column)"
  if diff <(grep -E "$EPS" "$A_OUT") <(grep -E "$EPS" "$C_OUT"); then echo "A vs C: IDENTICAL"; else echo "A vs C: DIFFERENT (see the diff above)"; fi
  echo "--- A vs C, WALL-CLOCK COLUMN STRIPPED (the numbers only)"
  if diff <(grep -E "$EPS" "$A_OUT" | sed "$STRIP") <(grep -E "$EPS" "$C_OUT" | sed "$STRIP"); then echo "A vs C (numbers only): IDENTICAL"; else echo "A vs C (numbers only): DIFFERENT"; fi
  echo "--- the control's own refusal count (must be 0: P58c)"
  grep -E "GATE RECORD" "$C_OUT"
  echo "--- A: the default path's gate-relevant lines (must be absent: the gate is off)"
  grep -cE "MEMORISATION RULE|GATE RECORD" "$A_OUT" | sed 's/^/A gate lines: /'
  echo "=== GATE SMOKE DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
