#!/bin/bash
# Build the honest-picture figure and copy it where Telegram can attach it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -c "import matplotlib" 2>/dev/null || echo "NOTE: doing the plot with hermes-env python (only place matplotlib 3.11 is installed)"
/home/yahya/hermes-env/bin/python3 -u lens/scripts/make_report_pic.py
cp -f lens/runs/HONEST-PICTURE.png /home/yahya/glassnet-honest-picture.png
ls -la /home/yahya/glassnet-honest-picture.png