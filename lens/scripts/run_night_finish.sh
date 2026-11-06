#!/bin/bash
# NIGHT FINISH — wait for the attention run, measure it, audit it, then STOP.
#
# This replaces the longer overnight supervisor on purpose: Yahya is shutting the PC down for the
# night, so starting the 3-hour merged-corpus run at ~21:50 and the ~10-hour matrix after it would
# only be killed at power-off. With `--resume` now verified, nothing is lost by stopping here: the
# remaining epochs continue tomorrow from the newest checkpoint.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
exec >> /tmp/night_finish.log 2>&1
echo "=== NIGHT FINISH START $(date '+%F %H:%M')"

while pgrep -f "[t]rain_chat.py .*--out t12_attn_d512" >/dev/null 2>&1; do sleep 30; done
echo "=== attention run ended $(date '+%H:%M')"
ATT=$(ls -t lens/runs/chat/t12_attn_d512_ep*.npz 2>/dev/null | head -1)
echo "--- latest attention ship: $ATT"
if [ -n "$ATT" ]; then
  echo "--- merged-corpus clean val (the common yardstick):"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -4
  echo "--- its own corpus (prepared_splitfix) clean val:"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -2
  echo "--- strict label-honesty audit:"
  TALK_CKPT="$(basename "$ATT")" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 \
      | grep -E "STRICT LABEL|s1\[talk\]" | tail -3
  {
    echo ""
    echo "### ATTENTION RUN — MEASURED (the token-order lever, first real test)"
    echo "Ship: $(basename "$ATT")  (trained with --attention on the splitfix corpus)"
    echo "Reference: t12_splitfix_d512_ep192 = 63.85% / ppl 85.4 on the same clean-val method."
    echo "Numbers measured above in this entry's run; see /tmp/night_finish.log for the raw lines."
  } >> LOG.md
fi
echo "=== NIGHT FINISH DONE $(date '+%H:%M') -- stopping here on purpose (Yahya is powering down)."
echo "    Tomorrow: resume the attention run from its newest checkpoint if incomplete, then"
echo "    merged-192, then the six-option matrix (all resumable now)."