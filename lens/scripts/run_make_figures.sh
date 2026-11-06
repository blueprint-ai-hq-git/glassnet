#!/bin/bash
# Regenerate every product figure from current results, then copy them somewhere attachable.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
/home/yahya/hermes-env/bin/python3 -u lens/figures.py
cp -f lens/runs/figures/*.png /home/yahya/ 2>/dev/null
ls -la /home/yahya/how_good_really.png /home/yahya/does_it_talk.png /home/yahya/what_works.png 2>&1