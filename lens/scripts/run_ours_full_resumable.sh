#!/bin/bash
# OUR METHOD, FULL ~16.9M-WINDOW CORPUS, RESUME-SAFE.  --out ours_d256_full_3r
#
# PREDICTIONS P83a-f, WRITTEN BEFORE THIS LINE WAS QUEUED (the project's rule; P82 is left free for the
# attention arm that P79f pre-committed to if the attention smoke passes):
#
#  P83a  THE RUN CHECKPOINTS MID-EPOCH. {out}_mid.pkl exists within 31 minutes of the first trained step and
#        is rewritten at least every ~30 minutes of training, each write named in the log. FALSIFIED IF no
#        mid file appears within 31 minutes, or a gap between writes exceeds 35 minutes of wall time.
#  P83b  AN INTERRUPTION COSTS AT MOST ONE INTERVAL. A stop (shutdown or SIGTERM) loses at most ~30 minutes
#        of steps, and the next launch continues from the recorded step (the log prints the RESUME-MID lines
#        with the step it continued from). FALSIFIED IF the next launch starts from step 1 while a mid
#        checkpoint exists, or if it silently refuses/resumes with a different shuffle order.
#  P83c  THE GATES STILL ACT. The gap gate refuses more than 0% of its checks and the agreement gate refuses
#        a substantial share, as measured on the attempt that was stopped today (16:50: gap 2,127/3,150 =
#        67.5% refused; agreement 1,765/4,925 = 35.8%; lens/runs/chat/ours_d256_full_3_gates.jsonl and
#        _agreement.jsonl). If a whole epoch runs with ~0% refusals, that is a finding AGAINST the gates.
#  P83d  THE CLAIM UNDER TEST (the run's actual question): the train-minus-held-out gap on the honest
#        by-position column comes in BELOW the old loop's 3.31 points on the same corpus
#        (lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt: train 38.57% / held-out 35.26% at epoch 3).
#        FALSIFIED IF the gap is at or above 3.31 points, or if the gap is "narrower" only because held-out
#        collapsed well under 35% (that would be buying the gap with accuracy and must be said plainly).
#  P83e  THIS RUN NEEDS 3+ MACHINE-DAYS. 10.8-13.8 h per epoch x 3 epochs = 33-41 h of GPU time at the
#        measured 1.33-1.70 steps/s (lens/runs/OURS-METHOD-COST.txt); the machine gives about 5.5 h a day.
#        FALSIFIED IF it finishes faster than that (which would mean the cost measurement was wrong).
#  P83f  THE TALK TEST STILL FAILS ON THIS ARM. The pad-free talk test on the best checkpoint is predicted
#        to stay incoherent (the same FAIL the mask-fixed twin produced). FALSIFIED IF it produces
#        prompt-relevant English, which would be the first time and would outrank everything else here.
#  NOT CLAIMED: nothing about attention (that is a separate arm), nothing shipped, no export, and no number
#  from this run is quoted without its file. A word-salad talk test means the run failed whatever the loss says.
#
# WHY THE SAME RECIPE, NEW HARNESS: the first attempt at this arm (run_ours_first.sh, 15:46 today) was
# launched with no mid-epoch checkpoint, and this turn measured what that costs: 10.8-13.8 h per epoch, saves
# only at an epoch boundary, machine stopped most nights -> that launch could not produce a single artifact.
# The recipe is identical (same step rule, gates, data, seed, flags); only the checkpointing changed, and the
# exactness of the resume was verified before this line was queued: lens/runs/MID-CKPT-RESUME.txt (P81a-f).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
TAG=ours_d256_full_3r
MID="lens/runs/chat/${TAG}_mid.pkl"
RESUME=""
if [ -f "$MID" ]; then
  RESUME="--resume-mid auto"
fi
{
  echo "=== OUR METHOD, FULL CORPUS, RESUME-SAFE ARM (P83a-f) $(date '+%F %H:%M')"
  echo "predictions: written in the header of lens/scripts/run_ours_full_resumable.sh before it was queued"
  echo "reference (OLD loop, same corpus, same cap): train 38.57% held-out 35.26% ppl 40.42, gap 3.31 points"
  echo "  source lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt (d=256, 16.9M windows, by-position column)"
  echo "cost measured this turn: 10.8-13.8 h per epoch -> 33-41 h for 3 epochs (lens/runs/OURS-METHOD-COST.txt)"
  echo "checkpoint every 1800 s; resume flag: '${RESUME:-none (starting fresh)}'"
  echo "mid checkpoint file: $MID"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method ours --out "$TAG" \
    --ckpt-seconds 1800 $RESUME
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/OURS-METHOD-RUN2-RESUMABLE.txt 2>&1
# LOG NAME: this runner's output goes to OURS-METHOD-RUN2-RESUMABLE.txt and NOT OURS-METHOD-RUN2.txt, which
# a parallel turn's lens/scripts/run_ours_3m.sh (the 3M-window arm, started 17:55) writes to. Two different
# experiments must never share one log file -- that is how this project's results got mixed before.
