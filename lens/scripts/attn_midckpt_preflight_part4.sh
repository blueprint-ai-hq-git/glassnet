#!/bin/bash
# attn_midckpt_preflight_part4.sh -- P85d: is --attention inside the resume SETTINGS FINGERPRINT?
# Same tiny config as part 3 (interval 0.05 s), same --out attnP85B, SAME mid checkpoint written by part 3 --
# but --attention DROPPED from the command line. Expected: exit non-zero with
# "RESUME-MID REFUSED: the settings on this command line are not the settings the checkpoint was written with".
# A resume that proceeded would mean a run can keep its name while its architecture changes underneath it.
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

{
echo ""
echo "--- PART 4 $(date '+%F %H:%M:%S') -- P85d REFUSAL: resume the attention checkpoint WITHOUT --attention"
echo -n "the mid checkpoint used: "; ls -l --time-style=+%H:%M:%S lens/runs/chat/attnP85B_mid.pkl 2>&1
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --ckpt-seconds 0.05 --resume-mid auto --out attnP85B > "$T/attnP85B_refuse.log" 2>&1
echo "refusal-test EXIT=$?  (non-zero = the fingerprint caught the architecture change)"
grep -n "RESUME-MID REFUSED" "$T/attnP85B_refuse.log" | head -3
echo "=== PART 4 DONE $(date '+%F %H:%M:%S')"
} >> "$R" 2>&1
