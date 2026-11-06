#!/bin/bash
# attn_midckpt_preflight.sh -- P85a-f: DOES THE MID-EPOCH CHECKPOINT + EXACT RESUME WORK WHEN --attention IS ON?
#
# Predictions P85a-f were written BEFORE this script was run:
#   lens/runs/ATTN-MIDCKPT-PREFLIGHT-PREDICTIONS-P85.txt
# Results append to:  lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt
# Per-run logs + checkpoints: lens/runs/_tmp/attnP85*.log and lens/runs/chat/attnP85*  (NEW TAGS ONLY)
#
# MIRRORS THE FLAG SHAPE OF lens/scripts/run_attn_full.sh (d=256 --window 32 --epochs 3 --batch 256 --lr 0.4
# --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end
# --direct --direct-mode tied --max-windows 16900000 --attention --ckpt-seconds 1800 --out
# ct_d256_full169_attn), scaled down to d=64 / window 8 / batch 64 / 6,400 windows / 1 epoch and with the
# interval shortened to 2 s. IT IS A SEPARATE FILE ON PURPOSE (two turns editing one runner is how
# experiments got mixed before); if run_attn_full.sh's flags change, change this mirror too.
#
# WHY THE HELPER SCRIPTS: lens/runs/_tmp/{mid_tools.py,cmp_npz.py} were written for the P81 verification and
# are reusable, but they must NEVER be edited (the P81 numbers came out of them).
#
# COST / PRIORITY: the 3M ours arm is training on cores 0-1 and the GPU right now. This pre-flight is pinned
# to cores 2-3, nice 19, ionice -c3, d=64 (a few tens of MB of VRAM), and runs ONE process at a time.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/lens/scripts:$PWD"   # chat_concepts/chat_data live in lens/scripts

T="lens/runs/_tmp"
R="lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt"
NEW=".venv-gpu/bin/python3 -u lens/scripts/train_chat.py"
RUN="taskset -c 2,3 nice -n 19 ionice -c3"
TINY="--d 64 --window 8 --batch 64 --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
--train-turn-end --direct --direct-mode tied"
ATT="--attention"

{
echo "================================================================================"
echo "=== ATTENTION MID-CKPT PRE-FLIGHT (P85a-f)  $(date '+%F %H:%M:%S')"
echo "predictions: lens/runs/ATTN-MIDCKPT-PREFLIGHT-PREDICTIONS-P85.txt (written BEFORE this run)"
echo "config: d=64 window=8 batch=64 max-windows=6400 (100 steps/epoch) seed=0 gpu, --attention, old-loop rule"
echo "cores 2-3, nice 19, ionice -c3; the 3M ours arm (cores 0-1) is the priority and must not stall"

# ---------------------------------------------------------------- P85a/c : control A
echo ""
echo "--- P85a/c CONTROL A: 1 epoch, --attention, NO checkpoint flags (--out attnP85A)"
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 --out attnP85A > "$T/attnP85A.log" 2>&1
echo "A EXIT=$?"
grep -E "^  ep 1/1 " "$T/attnP85A.log" | head -1
echo "-- the tensors this checkpoint actually carries (count, then the keys):"
python3 - <<'PY'
import numpy as np
z = np.load("lens/runs/chat/attnP85A_best.npz", allow_pickle=True)
print("   attnP85A_best.npz tensors: %d" % len(z.files))
print("   keys:", sorted(z.files))
PY
echo "   (for comparison, a non-attention d=256 ship carries 7 tensors per GRADE-P81.txt P81d)"

# ---------------------------------------------------------------- P85a : interrupted B
echo ""
echo "--- P85a INTERRUPTED B: same run WITH --ckpt-seconds 2 (--out attnP85B), killed mid-epoch on purpose"
rm -f lens/runs/chat/attnP85B_mid.pkl
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 --ckpt-seconds 2 --out attnP85B > "$T/attnP85B.log" 2>&1 &
BGPID=$!
python3 "$T/mid_tools.py" wait lens/runs/chat/attnP85B_mid.pkl 1 10 300
echo "wait exit=$?  (background pid $BGPID)"
kill -TERM $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
echo "killed; state on disk:"
python3 "$T/mid_tools.py" mid lens/runs/chat/attnP85B_mid.pkl
ls -l lens/runs/chat/attnP85B_mid.pkl lens/runs/chat/attnP85B_ep1_rngstart.json 2>&1
python3 "$T/mid_tools.py" hasline "MID-EPOCH CHECKPOINT" "$T/attnP85B.log"
echo "P85a exit=$?"

# ---------------------------------------------------------------- P85b/c : resume B
echo ""
echo "--- P85b/c RESUME B: --resume-mid auto, let it finish epoch 1"
$RUN $NEW $TINY $ATT --epochs 1 --max-windows 6400 --ckpt-seconds 2 --resume-mid auto --out attnP85B >> "$T/attnP85B.log" 2>&1
echo "B resume EXIT=$?"
python3 "$T/mid_tools.py" epoch "$T/attnP85A.log" "$T/attnP85B.log"
echo "P85b exit=$?"
python3 "$T/mid_tools.py" hasline "RESUME-MID: EPOCH 1 continues at window" "$T/attnP85B.log"
python3 "$T/mid_tools.py" hasline "RESUME-MID: " "$T/attnP85B.log"
echo "-- P85c: weights, control A vs interrupted+resumed B (attention on both)"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_ep1.npz lens/runs/chat/attnP85B_ep1.npz
echo "P85c-a exit=$?"
python3 "$T/cmp_npz.py" lens/runs/chat/attnP85A_best.npz lens/runs/chat/attnP85B_best.npz
echo "P85c-b exit=$?"

# ---------------------------------------------------------------- P85d : refusal on a changed flag
echo ""
echo "--- P85d REFUSAL: resume the SAME checkpoint with --attention DROPPED (expected NON-ZERO exit)"
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --ckpt-seconds 2 --resume-mid auto --out attnP85B > "$T/attnP85B_refuse.log" 2>&1
echo "refusal-test EXIT=$?  (non-zero = the fingerprint caught the architecture change)"
grep -n "RESUME-MID REFUSED" "$T/attnP85B_refuse.log" | head -3

echo ""
echo "=== PRE-FLIGHT DONE $(date '+%F %H:%M:%S')"
echo "=== READ THE EXIT CODES ABOVE, NOT THE INTENTION. d=64 numbers are wiring evidence only."
} >> "$R" 2>&1
