#!/bin/bash
# run_tied_d512_ext6.sh — d=512, FULL 16.9M-window corpus, SIX epochs: the capacity test CONTINUED.
#
# WHY (numbers, not taste): lens/runs/TIED-D512-FULL.txt lines 63-68 — the stored 3-epoch arm
# (ct_d512_full169_fix3) raised held-out by position at every epoch (33.17 / 35.61 / 37.18) and lowered
# held-out PPL at every epoch (38.97 / 32.37 / 29.35) while train accuracy FELL from ep2 to ep3. It was
# still improving when it stopped. 37.18% is a lower bound, not a ceiling. The only longer run in the
# project (ct_d256_6M_6ep, 6 epochs) peaked at ep5 and fell at ep6, so six is the length that has shown a peak.
#
# PREDICTIONS: lens/runs/D512-EXT6-PREDICTIONS-P94.txt (P94a-g), written BEFORE this file was queued.
# New --out tag ct_d512_full169_ext6 and new log lens/runs/TIED-D512-EXT6.txt: nothing existing is overwritten,
# and the stored 3-epoch arm stays the standalone record it is.
# Declared confound (also in the predictions): 6 epochs with --lr-decay 0.5 stretches the anneal, so ep2/ep3
# run at a higher lr than the stored arm's. ep1 is the near-reproduction (same lr 0.3, same seed).
# CHAINED ON PURPOSE: the pad-free DECIDER runs on the new best checkpoint in the SAME job, so a shutdown
# cannot leave the arm trained with nothing measuring whether it talks (the run_unk_drop.sh pattern).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

TAG=ct_d512_full169_ext6
BEST="lens/runs/chat/${TAG}_best.npz"
TWIN_BEST="lens/runs/chat/ct_d512_full169_fix3_best.npz"
LOG="lens/runs/TIED-D512-EXT6.txt"

{
  echo "=============================================================================================="
  echo "=== d=512 FULL CORPUS 16,900,000 windows, 6 EPOCHS — $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/D512-EXT6-PREDICTIONS-P94.txt (P94a-g, written before this was queued)"
  echo "extends: lens/runs/TIED-D512-FULL.txt (3 epochs, best ep3 37.18% by position, ppl still falling)"
  echo "declared confound: 6-epoch anneal (lr 0.3 -> 0.15 over 6 epochs) so ep2/ep3 lr is higher than the stored arm's"
  echo "=============================================================================================="

  t0=$(date +%s)
  taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 512 --window 32 --epochs 6 --batch 128 --lr 0.3 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --ckpt-seconds 1800 --out ${TAG}
  rc=$?
  echo "ARM EXIT=$rc   wall=$(( $(date +%s) - t0 ))s   $(date '+%F %H:%M:%S')"

  echo "--- $(date '+%F %H:%M:%S') epoch table from this run's own history (never the last checkpoint by default)"
  nice -n 19 ionice -c3 python3 lens/scripts/audit_memorisation.py > /dev/null 2>&1
  echo "  (full table: lens/runs/MEMORISATION-AUDIT.txt)"

  if [ ! -f "$BEST" ]; then
    echo "NO BEST CHECKPOINT: $BEST missing -> the DECIDER cannot run on the artifact that would ship."
    echo "NOT substituting the last checkpoint (project rule: always the best). Recorded as producing nothing."
    echo "=== CHAIN STOPPED $(date '+%F %H:%M:%S') ==="
    exit 7
  fi
  read -r BEST_EP BEST_ASST BEST_LEGACY BEST_LEGACY_FRAC < <(.venv-gpu/bin/python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("lens/runs/chat/ct_d512_full169_ext6_history.jsonl") if l.strip()]
b = max(rows, key=lambda r: r["val_asst_acc"])
print(b["epoch"], f'{b["val_asst_acc"]*100:.2f}', f'{b["val_masked_acc"]*100:.2f}', f'{b["val_masked_acc"]:.4f}')
PY
)
  echo "BEST epoch by BY-POSITION held-out: ep${BEST_EP} (${BEST_ASST}%; LEGACY by-token-id ${BEST_LEGACY}%)"
  echo "  (the legacy value is passed to the instrument as a FRACTION, ${BEST_LEGACY_FRAC}: BUG-TALKFAIR-LOGGED-UNITS)"

  echo "--- $(date '+%F %H:%M:%S') THE DECIDER: pad-free talk, this arm and the stored d512 twin in ONE run"
  export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-D512EXT6.txt"
  export CKPT_LIST="d512 EXT6 best ep${BEST_EP} (by-position ${BEST_ASST}%)=$BEST;STORED d512 3-epoch best ep3 (by-position 37.18%)=$TWIN_BEST"
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  echo "talk_fair_v3_ship.py exit=$?"

  echo "--- $(date '+%F %H:%M:%S') free-running by step + context ablation (P65c/P65d instrument)"
  TALK_FAIR_CKPT="$PWD/$BEST" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-D512EXT6.txt" \
  TALK_FAIR_LOGGED="$BEST_LEGACY_FRAC" \
  TALK_FAIR_EP="ep${BEST_EP} (BEST by-position ${BEST_ASST}%; legacy by-id ${BEST_LEGACY}%)" \
  TALK_FAIR_PREDICTIONS="lens/runs/D512-EXT6-PREDICTIONS-P94.txt (P94f)" \
    nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "talk_fair_test.py exit=$?"

  echo "=== CHAIN DONE $(date '+%F %H:%M:%S') ===  files: TIED-D512-EXT6.txt, MEMORISATION-AUDIT.txt,"
  echo "    PADFREE-TALK-D512EXT6.txt, TALK-FAIR-D512EXT6.txt. P94f decides whether ANYTHING ships."
} >> "$LOG" 2>&1
