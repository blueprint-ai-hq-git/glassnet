#!/bin/bash
# PER-CONCEPT ARM: matched pair in one run - shipped shared linear filter (control) then one nonlinear detector per concept.
# Same seed, same budget, same trainer. Scored with BOTH honesty instruments because they disagree (9 marginal vs 6 conditional
# on the shipped model) and quoting only the flattering one is the dishonesty this project exists to avoid.
# Baseline to beat: 9 of 14 marginal and 6 of 14 conditional at 80.78%. No machine paths here - this file is published.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
PY="${GLASSNET_PY:-python3}"
echo "=== per-concept arm (control then treatment) ==="; date '+%H:%M:%S'
"$PY" tools/train_per_concept.py 2>&1
date '+%H:%M:%S'
