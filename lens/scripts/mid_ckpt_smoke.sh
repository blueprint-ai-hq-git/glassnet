#!/bin/bash
# mid_ckpt_smoke.sh -- the verification for the mid-epoch checkpoint + resume added to
# lens/scripts/train_chat.py this turn. Predictions P81a-f were written BEFORE the code:
# lens/runs/MID-CKPT-RESUME-PREDICTIONS-P81.txt. This script only APPENDS to
# lens/runs/MID-CKPT-RESUME.txt and writes its per-run logs under lens/runs/_tmp/.
#
# DEVICE: gpu, deliberately, and tiny. The first attempt ran these on CPU and was dominated by the
# validation pass (the corpus has 368,369 held-out windows and the val loop walks all of them on every
# epoch) -- one tiny 20-step run took minutes. This smoke is d=64 / 6,400 windows / batch 64: a few tens of
# MB of VRAM against the trainer's ~1.1 GB, nice -19 and ionice -c3 so the full-corpus run keeps priority,
# and the runs are strictly sequential (one at a time), which is the rule this project already learned the
# hard way.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/lens/scripts:$PWD"   # chat_concepts/chat_data live in lens/scripts; the _tmp copy of
                                             # the trainer is not next to them and cannot find them otherwise
                                             # (measured: ModuleNotFoundError: No module named 'chat_concepts')

T="lens/runs/_tmp"
R="lens/runs/MID-CKPT-RESUME.txt"
TINY="--d 64 --window 8 --batch 64 --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
--train-turn-end --direct --direct-mode tied --method ours"
PRE=".venv-gpu/bin/python3 -u lens/runs/_tmp/train_chat_prefix3.py"
NEW=".venv-gpu/bin/python3 -u lens/scripts/train_chat.py"
RUN="nice -n 19 ionice -c3"

{
echo "================================================================"
echo "=== MID-CKPT SMOKE $(date '+%F %H:%M:%S')  (P81a-f; predictions: MID-CKPT-RESUME-PREDICTIONS-P81.txt)"
echo "config: $TINY --max-windows 6400   (100 steps/epoch)"

# ---------------------------------------------------------------- P81a
echo ""
echo "--- P81a: DEFAULT PATH UNCHANGED (pre-change copy vs changed script, neither using the new flags)"
$RUN $PRE $TINY --epochs 1 --max-windows 6400 --out _midctl_pre > "$T/_midctl_pre.log" 2>&1
echo "prefix3 EXIT=$?"
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --out _midctl_new > "$T/_midctl_new.log" 2>&1
echo "changed EXIT=$?"
python3 "$T/mid_tools.py" epoch "$T/_midctl_pre.log" "$T/_midctl_new.log"
echo "P81a exit=$?"

# ---------------------------------------------------------------- P81b/c/d
echo ""
echo "--- P81c/d CONTROL: uninterrupted 1-epoch run (--out _midA), no new flags"
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --out _midA > "$T/_midA.log" 2>&1
echo "A EXIT=$?"

echo "--- P81b: same run WITH --ckpt-seconds 2 (--out _midB), KILLED mid-epoch on purpose"
rm -f lens/runs/chat/_midB_mid.pkl
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --ckpt-seconds 2 --out _midB > "$T/_midB_run1.log" 2>&1 &
BGPID=$!
python3 "$T/mid_tools.py" wait lens/runs/chat/_midB_mid.pkl 1 10 300
echo "wait exit=$?  (backgound pid $BGPID)"
kill -TERM $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
echo "killed; state on disk:"
python3 "$T/mid_tools.py" mid lens/runs/chat/_midB_mid.pkl
ls -l lens/runs/chat/_midB_mid.pkl lens/runs/chat/_midB_ep1_rngstart.json 2>&1
python3 "$T/mid_tools.py" hasline "MID-EPOCH CHECKPOINT" "$T/_midB_run1.log"
echo "P81b exit=$?"

echo "--- P81c/d: RESUME that run (--resume-mid auto) and let it finish epoch 1"
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --ckpt-seconds 2 --resume-mid auto --out _midB >> "$T/_midB_run1.log" 2>&1
echo "B resume EXIT=$?"
python3 "$T/mid_tools.py" epoch "$T/_midA.log" "$T/_midB_run1.log"
echo "P81c exit=$?"
python3 "$T/mid_tools.py" hasline "RESUME-MID: EPOCH 1 continues at window" "$T/_midB_run1.log"
echo "P81e-a exit=$?"
echo "-- P81d: weights, control A vs interrupted+resumed B"
python3 "$T/cmp_npz.py" lens/runs/chat/_midA_ep1.npz lens/runs/chat/_midB_ep1.npz
echo "P81d-a exit=$?"
python3 "$T/cmp_npz.py" lens/runs/chat/_midA_best.npz lens/runs/chat/_midB_best.npz
echo "P81d-b exit=$?"

# ---------------------------------------------------------------- P81e (refusal)
echo ""
echo "--- P81e: a resume whose settings differ must be REFUSED (--lr 0.5 vs the checkpoint's 0.4)"
$RUN $NEW $TINY --epochs 1 --max-windows 6400 --lr 0.5 --ckpt-seconds 2 --resume-mid auto --out _midB > "$T/_midB_refuse.log" 2>&1
echo "refusal-test EXIT=$?  (expected NON-ZERO)"
grep -n "RESUME-MID REFUSED" "$T/_midB_refuse.log" | head -3
echo "P81e-b done"

# ---------------------------------------------------------------- P81f
echo ""
echo "--- P81f: best-checkpoint memory across an interruption INSIDE EPOCH 2"
echo "-- control: 2 epochs uninterrupted (--out _midA2)"
$RUN $NEW $TINY --epochs 2 --max-windows 6400 --out _midA2 > "$T/_midA2.log" 2>&1
echo "A2 EXIT=$?"
echo "-- interrupted: 2 epochs WITH --ckpt-seconds 2 (--out _midB2), killed inside epoch 2"
rm -f lens/runs/chat/_midB2_mid.pkl
$RUN $NEW $TINY --epochs 2 --max-windows 6400 --ckpt-seconds 2 --out _midB2 > "$T/_midB2.log" 2>&1 &
BGPID2=$!
python3 "$T/mid_tools.py" wait lens/runs/chat/_midB2_mid.pkl 2 10 600
echo "wait exit=$?  (background pid $BGPID2)"
kill -TERM $BGPID2 2>/dev/null
wait $BGPID2 2>/dev/null
python3 "$T/mid_tools.py" mid lens/runs/chat/_midB2_mid.pkl
echo "-- resumed: finish epoch 2"
$RUN $NEW $TINY --epochs 2 --max-windows 6400 --ckpt-seconds 2 --resume-mid auto --out _midB2 >> "$T/_midB2.log" 2>&1
echo "B2 resume EXIT=$?"
python3 "$T/mid_tools.py" epoch "$T/_midA2.log" "$T/_midB2.log"
echo "P81f-lines exit=$?"
python3 "$T/mid_tools.py" hasline "best held-out before the interruption" "$T/_midB2.log"
echo "P81f-memory-line exit=$?"
echo "-- P81f: the shipped best checkpoint must be the same file either way"
python3 "$T/cmp_npz.py" lens/runs/chat/_midA2_best.npz lens/runs/chat/_midB2_best.npz
echo "P81f exit=$?"
echo ""
echo "=== MID-CKPT SMOKE DONE $(date '+%F %H:%M:%S')"
echo "=== ARTIFACTS: lens/scripts/mid_ckpt_smoke.sh, $T/_mid*.log, this file. Nothing deleted."
} >> "$R" 2>&1
