#!/bin/bash
# attn_midckpt_preflight_part3.sh -- P85a/b/c WITH THE INTERVAL CORRECTED.
#
# WHAT WENT WRONG IN THE FIRST TWO ATTEMPTS (mine, not the trainer's):
#   --ckpt-seconds 2 was LONGER THAN THE WHOLE TINY TRAINING LOOP. At d=64 / batch 64 / 6,400 windows the
#   100-step loop runs in about a second; the ~98-104 s of the epoch line is almost all the VALIDATION pass
#   (368,369 held-out windows in this corpus). So (perf_counter() - last_ckpt) never reached 2.0 and the
#   mid-ckpt branch at train_chat.py:1085 never fired -- no MID-EPOCH line, no _mid.pkl, and the resume then
#   exited 1 with "no mid-epoch checkpoint". Two runs produced that, both recording it in
#   lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt. It is the same class of error as the P79c band: a harness quantity
#   scaled wrongly against the thing it is supposed to exercise, and the fix is a smaller interval, not a
#   different architecture. THIS IS NOT A FINDING ABOUT ATTENTION and is recorded as a harness mistake.
#   POSITIVE EVIDENCE the mechanism was never reached: both B attempts wrote {out}_ep1_rngstart.json, which
#   is written on the SAME condition (line 906), while a 2 s gap never occurred inside the 1 s loop.
#
# Also stated: the P81 verification (which DID see a mid checkpoint at step 10 after 103 s) ran --method ours,
# where every step costs ~10 s because the gates re-score the probe; that is why 2 s was enough there.
#
# THIS FILE: interval 0.05 s (about 20-40 writes per 100-step loop), everything else identical.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/lens/scripts:$PWD"

T="lens/runs/_tmp"
R="lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt"
NEW=".venv-gpu/bin/python3 -u lens/scripts/train_chat.py"
RUN="taskset -c 2,3 nice -n 19 ionice -c3"
TINY="--d 64 --window 8 --batch 64 --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
--train-turn-end --direct --direct-mode tied"
ATT="--attention"
IV="--ckpt-seconds 0.05"

{
echo ""
echo "--- PART 3 $(date '+%F %H:%M:%S') -- SAME TEST, INTERVAL CORRECTED TO 0.05 s (the 2 s in parts 1-2 was"
echo "--- longer than the whole 100-step tiny loop, so the mid-ckpt branch was never reached: my harness error)"

echo ""
echo "--- P85a INTERRUPTED B: --ckpt-seconds 0.05 (--out attnP85B), killed once the mid pkl is honest"
rm -f lens/runs/chat/attnP85B_mid.pkl
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 $IV --out attnP85B > "$T/attnP85B.log" 2>&1 &
BGPID=$!
python3 "$T/mid_tools.py" wait lens/runs/chat/attnP85B_mid.pkl 1 10 240
echo "wait exit=$?  (background pid $BGPID)"
python3 "$T/mid_tools.py" mid lens/runs/chat/attnP85B_mid.pkl
kill -TERM $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
echo "killed; state on disk:"
ls -l --time-style=+%H:%M:%S lens/runs/chat/attnP85B_mid.pkl lens/runs/chat/attnP85B_ep1_rngstart.json 2>&1
echo -n "MID-EPOCH CHECKPOINT lines in the log (a large count proves the 100-step loop is ~1 s long): "
grep -c "MID-EPOCH CHECKPOINT" "$T/attnP85B.log"
python3 "$T/mid_tools.py" hasline "MID-EPOCH CHECKPOINT" "$T/attnP85B.log"
echo "P85a exit=$?"

echo ""
echo "--- P85b/c RESUME B: --resume-mid auto, same flags, let it finish epoch 1"
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 $IV --resume-mid auto --out attnP85B >> "$T/attnP85B.log" 2>&1
echo "B resume EXIT=$?"
python3 "$T/mid_tools.py" hasline "RESUME-MID: EPOCH 1 continues at window" "$T/attnP85B.log"
python3 "$T/mid_tools.py" epoch "$T/attnP85A.log" "$T/attnP85B.log"
echo "P85b exit=$?"
echo "-- P85c: weights, control A vs interrupted+resumed B, both with --attention"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_ep1.npz lens/runs/chat/attnP85B_ep1.npz
echo "P85c-a exit=$?"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_best.npz lens/runs/chat/attnP85B_best.npz
echo "P85c-b exit=$?"
echo "=== PART 3 DONE $(date '+%F %H:%M:%S')"
} >> "$R" 2>&1
