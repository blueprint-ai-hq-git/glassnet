#!/bin/bash
# TALK-QUALITY CONTROL LADDER — does the v2 meter actually separate garbage from real text?
# A meter that cannot fail on known-bad output is decorative. ep1 is a known-garbage model (its
# samples are visibly junk), ep8 is barely trained, and the shipped ship is the current best.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec >> /tmp/talk_quality_controls.log 2>&1
echo "=== CONTROL LADDER START $(date '+%F %H:%M')"
echo "########## EP1 (known garbage -- MUST FAIL the bar)"
bash lens/scripts/run_talk_quality.sh lens/runs/chat/ab_prepared_ext_ep1.npz EP1-control "$PWD/data/chat/prepared_ext"
echo "########## EP8 (barely trained)"
bash lens/scripts/run_talk_quality.sh lens/runs/chat/ab_prepared_ext_ep8.npz EP8-control "$PWD/data/chat/prepared_ext"
echo "########## SHIP t12_splitfix_d512_ep192 (current best, own corpus)"
bash lens/scripts/run_talk_quality.sh lens/runs/chat/t12_splitfix_d512_ep192.npz SHIP-splitfix "$PWD/data/chat/prepared_splitfix"
echo "=== CONTROL LADDER DONE $(date '+%F %H:%M')"