#!/bin/bash
# (1) numpy for the data venv so the builder can run; (2) speech_commands from its OFFICIAL source:
# HF's repo only carries a loading script (verified: 3 files: README/.gitattributes/.py), so the
# dataset itself comes from Google's published archive (CC BY 4.0). Download runs in background.
set -e
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-data/bin/pip install --quiet numpy
.venv-data/bin/python3 -c "import numpy; print('numpy', numpy.__version__, 'in .venv-data')"