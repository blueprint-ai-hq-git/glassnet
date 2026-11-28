#!/bin/bash
# DECORRELATED-LABEL ARM: a matched pair in one run - control (all 14 concepts) then the same trainer on a label set with every
# pair above 0.80 correlation trimmed away. Same seed, same budget, same code path, so the difference cannot come from any of them.
# Baseline to beat: 9 of 14 honest at 80.78% accuracy. No machine paths here - this file is published.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
PY="${GLASSNET_PY:-python3}"
echo "=== decorrelated-label arm (control then treatment) ==="; date '+%H:%M:%S'
"$PY" tools/train_decorrelated.py 2>&1
date '+%H:%M:%S'
