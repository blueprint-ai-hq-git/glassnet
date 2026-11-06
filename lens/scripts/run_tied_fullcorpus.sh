#!/bin/bash
# CRITICAL PATH 1 OF 2, UNCAPPED: EVERY TRAINING WINDOW (16,878,132), d=256, 3 epochs, best-val checkpointing.
# RE-QUEUED 2026-10-22 16:4x BY THE SELF-WAKE, AFTER ITS FIRST UNCAPTED LAUNCH WAS OOM-KILLED:
#   glassnet-q-2591701866.scope, "Result: oom-kill" at 16:13:56, "Mem peak: 10G (swap: 2.7G)", duration 3min 56s,
#   dead inside the data build before its first epoch line. P63b ("peak RSS below 9.0 GB") is therefore a FAIL:
#   MEASURED 10G + 2.7G swap, not <9.0. THE CAUSE WAS THE WIDTH OF THE CONTAINERS, not the sampler: X alone was
#   int64 = 4.32 GB at this window count, and three separate float/bool masks added 2.6 GB.
# THE FIX (values proven untouched, lens/runs/MEMFIX2-PREDICTIONS.txt P64a-e, written BEFORE the edit):
#   X, Y int64 -> int32 (every token id in this corpus is 0..12001: lens/runs/MEMFIX2-TEST.txt P64e) and
#   is_asst/is_turnend float32 -> bool. Verified: values identical on 12/12 cases (MEMFIX2-TEST.txt P64a);
#   the mask's own printed numbers identical, 614/1,197 assistant targets and 15 turn-end targets, epoch lines
#   digit-for-digit on the real corpus (lens/runs/MEMFIX2-MASK.txt P64b); int32 vs int64 windows give
#   BIT-IDENTICAL logits, loss and all 6 gradient tensors on both CPU and GPU (lens/runs/D-TYPE-EQUIV.txt P64c).
#   Pre-change copies kept: lens/runs/_tmp/chat_data_prefix_memfix2.py, train_chat_prefix_memfix2.py.
#
# WHY THIS LINE EXISTS. The capped arm (6,000,000 windows, lens/scripts/run_tied_alldata.sh) is a hand-edit:
# the machine could not build the full window set. Measured 2026-10-22: RSS reached 9.0 GB and was still
# climbing at 16.9M windows (the d=512 variant peaked at 11.2 GB) and both attempts were stopped by hand before
# the kernel killed them.
#
# THE CAUSE WAS FOUND AND FIXED THIS TURN. lens/scripts/chat_data.py::make_windows allocated the (N, window)
# INDEX array and the window array at the same time, in int64: at N = 16,878,132 and window = 32 that is
# 4.32 GB of indices PLUS 4.32 GB of windows PLUS the 1.18 GB corpus. It now gathers in bounded chunks
# (<= 2^23 cells), so the transient index array is 67 MB, and the returned values are BIT-IDENTICAL to the old
# ones: lens/scripts/memfix_check.py, 12/12 cases over two window sizes, both the capped and uncapped path,
# every element of X and Y and the window starts compared - lens/runs/MEMFIX-TEST.txt. Nothing is deleted.
# Pre-change copy kept: lens/runs/_tmp/chat_data_prefix_memfix.py.
# PREDICTED PEAK RSS FOR THIS RUN, by arithmetic on those measured parts: X 4.32 GB + corpus 1.18 GB + the two
# assistant-mask arrays 2.36 GB + targets 0.2 GB = ~8.1 GB on a 14.75 GB machine whose desktop holds 2-3 GB.
# THAT PREDICTION IS A MEASURED FAIL (P63b): the 16:13 launch was OOM-killed at a 10G peak. The parts are now
# X 2.16 GB (int32) + is_asst 0.29 GB + is_turnend 0.29 GB (bool) + ids 2.35 GB + Wtr 0.07 GB = ~5.3 GB of
# arrays, i.e. ~6.5 GB with the interpreter and the desktop - which is what P64d predicts (6.5 GB or lower).
#
# PREDICTIONS, written BEFORE this line is queued (the three original ones are unchanged from 2026-10-22 15:40):
#  1. if the undertraining explanation is right, held-out beats the 1.5M-window tied run's peak 29.71% and the
#     talk test improves.
#  2. if held-out stays in the 27-30% band, data volume was NOT the binding constraint and the next test is
#     capacity (d=512).
#  3. the train/held-out gap is expected to stay wide; that is the memorisation question, measured here and
#     attacked separately, not hidden by the gate.
#  P63a this arm reaches epoch 1 and writes its first per-epoch checkpoint. Three earlier uncapped attempts died
#       inside make_windows before any epoch line (lens/runs/TIED-ALLDATA.txt, lens/runs/TIED-D512.txt).
#  P63b peak RSS stays below 9.0 GB.
#  P63c the first epoch's wall time is 2.8-4.0x the capped arm's first epoch (the window count is 2.81x). Slower
#       than 6x means something other than the window count is the bottleneck, and that is the finding.
#  P63d the masked held-out number is reported from the DISJOINT val split only (87,035 windows, 0 overlap with
#       train: lens/runs/VAL-TRAIN-LEAK-mix.txt) and it names that file, or it is not reported at all.
#  ADDED 2026-10-22 16:4x, before this line was re-queued (the P63 battery above still stands, P63b as a FAIL):
#  P64d peak RSS 6.5 GB or LOWER, read from this arm's own scope: `systemctl --user show -p MemoryPeak
#       glassnet-q-<unit>.scope`. If it is again near 10G, the width fix was not the binding term and that is
#       the finding - the next step would be to stop holding all windows in RAM at once.
#  P64g the FIRST EPOCH LINE is written this time. Three earlier uncapped attempts died inside the data build
#       with no epoch line at all (TIED-ALLDATA.txt, TIED-D512.txt, TIED-FULLCORPUS.txt).
#  P64h THE TALK TEST IS STILL WORD SALAD. Predicted from the capped arm: 6,000,000 windows (3.5x the earlier
#       1.5M) raised masked held-out to 34.92% and its talk section was still degenerate
#       (lens/runs/TIED-ALLDATA.txt, "CAN IT TALK?"). 2.8x more data on the same recipe is not expected to fix
#       generation on its own, and the run is not to be called progress on its loss - only on its English.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS (16,878,132 windows, UNCAPPED) d=256 3 epochs $(date '+%F %H:%M')"
  echo "reference: tied/1.5M-windows peaked 29.71% (ep4), ended 27.37%, train acc 44.14%, talk still salad"
  echo "chunked window sampler verified bit-identical: lens/runs/MEMFIX-TEST.txt (12/12 cases)"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 6 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 6000000 --out ct_d256_6M_6ep
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/TIED-FULLCORPUS.txt 2>&1
