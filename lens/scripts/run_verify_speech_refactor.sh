#!/bin/bash
# VERIFY the speech refactor: re-extract the features through lens/speech.py (the new single
# implementation) and require the result to be IDENTICAL to the dataset built before the refactor.
# A "moved code, same maths" claim must be proved, not asserted.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
cp data/speech/speech_v1.npz /tmp/speech_v1_before.npz
sha256sum /tmp/speech_v1_before.npz
.venv-gpu/bin/python3 -u lens/scripts/speech_data.py > /tmp/speech_data_refactor.log 2>&1
echo "RC=$?"
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np
a = np.load("/tmp/speech_v1_before.npz", allow_pickle=True)
b = np.load("data/speech/speech_v1.npz", allow_pickle=True)
same_X = np.array_equal(a["X"], b["X"])
same_y = np.array_equal(a["y"], b["y"])
same_s = np.array_equal(a["split"], b["split"])
print(f"  X identical: {same_X}   y identical: {same_y}   split identical: {same_s}")
print(f"  max abs feature difference: {float(np.abs(a['X'] - b['X']).max()):.3e}")
print("  VERDICT:", "PASS - refactor is maths-identical" if (same_X and same_y and same_s)
      else "FAIL - the refactor changed the features")
PY