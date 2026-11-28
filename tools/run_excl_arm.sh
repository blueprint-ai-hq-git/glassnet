#!/bin/bash
# EXCLUSIVITY ARM: train the shipped 12-line path with the concept-exclusivity term on, then run the honesty gate on the result.
# Baseline to beat: 9 of 14 concept neurons honest on the shipped model.
#
# NO machine paths in this file. It is published, so the interpreter comes from GLASSNET_PY if set and falls back to python3 on
# PATH, and the project root is resolved from this script's own location. A hardcoded home directory or disk label in a public file
# is a privacy leak, not a convenience.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
PY="${GLASSNET_PY:-python3}"
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
