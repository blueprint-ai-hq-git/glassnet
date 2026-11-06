#!/bin/bash
# agreement_smoke.sh -- the smoke test for PLAN item (d2d), part 1: THE AGREEMENT GATE (LEARN-NOT-MEMORISE.md
# section 2b) WIRED INTO THE SHIPPED TRAINER. Predictions first: lens/runs/AGREEMENT-GATE-PREDICTIONS.txt
# (P60a, P60b, P60c, P60m).
#
# FOUR invocations on the SMALL corpus (data/chat/prepared_gate_small), 2 epochs, CPU, nice 19:
#   A  the PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix_agree.py, md5 9ff15c00cfde6f7b99b1e0866408c3c0
#      = the shipped file exactly as it was before this turn), default path        -> the reference
#   B  the CHANGED script, NO new flags                                          -> P60a: A vs B
#   C  the CHANGED script, --gates agreement --agree-variant off (the CONTROL)    -> P60b: A vs C, refuses 0
#   D  the CHANGED script, --gates agreement (the DEFAULT detector, vs_probe)     -> the numbers + P60m
#
# Each arm's output goes to its own file in lens/runs/_tmp/ and is then appended to lens/runs/AGREEMENT-SMOKE.txt,
# so a re-run cannot corrupt the comparison. The comparison is made twice: on the whole epoch line, and with the
# trailing wall-clock column stripped (the gate arms are slower by construction and a bar that failed on the
# clock would be failing on the clock, not on the numbers).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small"
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/AGREEMENT-SMOKE.txt
A_OUT=lens/runs/_tmp/agree_smoke_A.txt
B_OUT=lens/runs/_tmp/agree_smoke_B.txt
C_OUT=lens/runs/_tmp/agree_smoke_C.txt
D_OUT=lens/runs/_tmp/agree_smoke_D.txt
TINY="--d 128 --window 32 --epochs 2 --batch 256 --lr 0.4 --lam-concept 0.3 --seed 0 --device cpu"
EPS='^  ep [0-9]+/'
STRIP='s/  ([0-9]*s)$//'

{
  echo "=== AGREEMENT SMOKE START $(date '+%F %H:%M:%S')  (per-arm output in lens/runs/_tmp/agree_smoke_{A,B,C,D}.txt)"
  echo "--- A: PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix_agree.py), default path"
  nice -n 19 ionice -c3 python3 -u lens/runs/_tmp/train_chat_prefix_agree.py $TINY --out _agree_smoke_A > "$A_OUT" 2>&1
  echo "A EXIT=$?"
  echo "--- B: CHANGED script, NO new flags (must be identical to A: P60a)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --out _agree_smoke_B > "$B_OUT" 2>&1
  echo "B EXIT=$?"
  echo "--- C: CHANGED script, --gates agreement --agree-variant off (CONTROL, must refuse 0: P60b)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates agreement \
    --agree-variant off --out _agree_smoke_C > "$C_OUT" 2>&1
  echo "C EXIT=$?"
  echo "--- D: CHANGED script, --gates agreement (DEFAULT detector vs_probe) - the numbers"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates agreement --out _agree_smoke_D > "$D_OUT" 2>&1
  echo "D EXIT=$?"
  echo "--- A vs B, WHOLE EPOCH LINE (P60a)"
  if diff <(grep -E "$EPS" "$A_OUT") <(grep -E "$EPS" "$B_OUT"); then echo "A vs B: IDENTICAL"; else echo "A vs B: DIFFERENT (see the diff above)"; fi
  echo "--- A vs C, WHOLE EPOCH LINE (P60b, includes the wall-clock column)"
  if diff <(grep -E "$EPS" "$A_OUT") <(grep -E "$EPS" "$C_OUT"); then echo "A vs C: IDENTICAL"; else echo "A vs C: DIFFERENT (see the diff above)"; fi
  echo "--- A vs C, WALL-CLOCK COLUMN STRIPPED (the numbers only)"
  if diff <(grep -E "$EPS" "$A_OUT" | sed "$STRIP") <(grep -E "$EPS" "$C_OUT" | sed "$STRIP"); then echo "A vs C (numbers only): IDENTICAL"; else echo "A vs C (numbers only): DIFFERENT"; fi
  echo "--- the control's own refusal count (must be 0: P60b)"
  grep -E "AGREEMENT RECORD" "$C_OUT"
  echo "--- D: the agreement numbers per step and the P60m equivalence line"
  grep -E "agreement step|P60m|AGREEMENT RECORD|agreement:" "$D_OUT"
  echo "--- A: the default path must carry NO agreement lines at all (the gate is off)"
  grep -cE "AGREEMENT RULE|AGREEMENT RECORD" "$A_OUT" | sed 's/^/A agreement lines: /'
  echo "--- per-step JSONL row counts (must equal the step counts: 10 for 2 epochs of 5 steps)"
  wc -l < lens/runs/chat/_agree_smoke_C_agreement.jsonl | sed 's/^/C rows: /'
  wc -l < lens/runs/chat/_agree_smoke_D_agreement.jsonl | sed 's/^/D rows: /'
  echo "=== AGREEMENT SMOKE DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
