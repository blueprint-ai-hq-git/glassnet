#!/bin/bash
# one_chat smoke: scripted input through the single-entry chat (text, image, audio, video).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cat <<'IN' | nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/one_chat.py
hello , how are you ?
/home/yahya/cam_frame13.png
/home/yahya/amar_slang_p1.ogg
/home/yahya/cam_fashion.mp4
exit
IN