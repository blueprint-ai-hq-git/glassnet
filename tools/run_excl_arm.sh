#!/bin/bash
# EXCLUSIVITY ARM: train the same 12-line path with the new concept-exclusivity term on, then run the honesty gate on the result.
# Baseline to beat: 9 of 14 concept neurons honest on the shipped model.
set -uo pipefail
cd /home/yahya/glassnet-public-prep || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
export GLASSNET_EXCL="${1:-0.5}"
echo "=== exclusivity arm, GLASSNET_EXCL=$GLASSNET_EXCL ==="; date '+%H:%M:%S'
"$PY" -c "
import sys; sys.path.insert(0,'.')
from glassnet.api import GlassNet
m=GlassNet('fashion-mnist', out_dir='runs/excl')
m.train(silent=True)
print('test accuracy: %.2f%%' % (m.accuracy()*100))
m.save('runs/excl/model.npz')
"
echo "--- honesty gate on the exclusivity model ---"
"$PY" tools/check_concept_honesty.py runs/excl/model.npz 2>&1 | tail -14
date '+%H:%M:%S'
