#!/bin/bash
# attn_midckpt_preflight_part2.sh -- CONTINUATION of lens/scripts/attn_midckpt_preflight.sh (P85a-f).
#
# WHY THIS FILE EXISTS, stated plainly: the first attempt ran as ONE foreground tool call and was killed by
# the harness's own 420-second cap in the middle of the interrupted arm (arm A had finished; the log
# lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt ends after the 'INTERRUPTED B' header). That is a HARNESS limit, not a
# finding about the trainer. Nothing was lost except wall time: arm A's checkpoint and log are on disk and are
# REUSED here as the control (lens/runs/chat/attnP85A_ep1.npz, _tmp/attnP85A.log) rather than re-run, so the
# comparison is still control-vs-interrupted with identical flags.
#
# This file runs the SAME flags as the original (see it for the mirror note against run_attn_full.sh) and
# appends to the SAME result file.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/lens/scripts:$PWD"

T="lens/runs/_tmp"
R="lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt"
NEW=".venv-gpu/bin/python3 -u lens/scripts/train_chat.py"
RUN="taskset -c 2,3 nice -n 19 ionice -c3"
TINY="--d 64 --window 8 --batch 64 --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
--train-turn-end --direct --direct-mode tied"
ATT="--attention"

{
echo ""
echo "--- CONTINUATION $(date '+%F %H:%M:%S') -- the first attempt was cut by the 420 s tool cap mid-arm-B,"
echo "--- so arm A is REUSED from disk as the control (its epoch line is above, unchanged)."

echo ""
echo "--- P85a INTERRUPTED B (re-run): --ckpt-seconds 2 --out attnP85B, killed as soon as the mid pkl is honest"
rm -f lens/runs/chat/attnP85B_mid.pkl
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 --ckpt-seconds 2 --out attnP85B > "$T/attnP85B.log" 2>&1 &
BGPID=$!
python3 "$T/mid_tools.py" wait lens/runs/chat/attnP85B_mid.pkl 1 10 240
echo "wait exit=$?  (background pid $BGPID)"
kill -TERM $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
echo "killed; state on disk:"
python3 "$T/mid_tools.py" mid lens/runs/chat/attnP85B_mid.pkl
ls -l lens/runs/chat/attnP85B_mid.pkl lens/runs/chat/attnP85B_ep1_rngstart.json 2>&1
python3 "$T/mid_tools.py" hasline "MID-EPOCH CHECKPOINT" "$T/attnP85B.log"
echo "P85a exit=$?"

echo ""
echo "--- P85b/c RESUME B: --resume-mid auto, let it finish epoch 1"
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 --ckpt-seconds 2 --resume-mid auto --out attnP85B >> "$T/attnP85B.log" 2>&1
echo "B resume EXIT=$?"
python3 "$T/mid_tools.py" epoch "$T/attnP85A.log" "$T/attnP85B.log"
echo "P85b exit=$?"
python3 "$T/mid_tools.py" hasline "RESUME-MID: EPOCH 1 continues at window" "$T/attnP85B.log"
echo "-- P85c: weights, control A vs interrupted+resumed B (attention on both)"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_ep1.npz lens/runs/chat/attnP85B_ep1.npz
echo "P85c-a exit=$?"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_best.npz lens/runs/chat/attnP85B_best.npz
echo "P85c-b exit=$?"
echo "=== CONTINUATION DONE $(date '+%F %H:%M:%S')"
} >> "$R" 2>&1
