#!/bin/bash
# CAPACITY ARM. The shipped model maps 784 pixels to 14 concepts with a SINGLE linear layer (concept_hidden = 0), while seven
# concept-label pairs correlate above 0.80 with each other. A linear readout cannot separate labels that share a direction, so the
# dishonest names may be a capacity limit rather than a training one. One variable: a hidden layer inside the concept head.
#
# Baseline to beat: 9 of 14 honest, 80.78% accuracy. No machine paths here - this file is published.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
PY="${GLASSNET_PY:-python3}"
HID="${1:-64}"
OUT="runs/caph${HID}"
export GLASSNET_EXCL="${2:-0}"
echo "=== capacity arm, concept_hidden=$HID, GLASSNET_EXCL=$GLASSNET_EXCL ==="; date '+%H:%M:%S'
"$PY" -c "
import sys; sys.path.insert(0,'.')
import glassnet.autopilot as ap
from glassnet.api import GlassNet
ap.AUTO_PILOT_DEFAULTS['concept_hidden'] = $HID      # one variable: give the concept head a hidden layer
m = GlassNet('fashion-mnist', out_dir='$OUT')
m.train(silent=True)
print('test accuracy: %.2f%%' % (m.accuracy()*100))
print('concept_hidden used:', int(getattr(m.model, 'concept_hidden', -1)))
m.save('$OUT/model.npz')
"
echo "--- honesty gate on the capacity model ---"
"$PY" tools/check_concept_honesty.py "$OUT/model.npz" 2>&1 | tail -12
date '+%H:%M:%S'
