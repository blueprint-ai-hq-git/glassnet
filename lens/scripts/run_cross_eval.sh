#!/bin/bash
# CROSS-CORPUS EVAL — the legal comparison that finally answers "are we worse, or is it the corpus?"
# Both models are scored in ONE vocabulary, so the numbers sit on the same scale.
# Usage: bash lens/scripts/run_cross_eval.sh <ship> <vocab-corpus-dir> <val-corpus-dir> <label>
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP="${1:?usage: run_cross_eval.sh <ship> <vocab-corpus> <val-corpus> <label>}"
VC="${2:?vocab corpus dir (the corpus the ship trained on)}"
LC="${3:?val corpus dir (the held-out text to measure)}"
LABEL="${4:-$(basename "${SHIP%.npz}")}"
TRAIN="${5:-}"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/cross_eval.py "$SHIP" "$VC" "$LC" "$LABEL" "$TRAIN"