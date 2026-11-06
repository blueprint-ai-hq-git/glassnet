#!/bin/bash
# run_medical_check.sh — verify the medical dataset loads and builds a task, BEFORE training on it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -35
import os, sys
import numpy as np
PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import vl_data

print("=" * 100)
print("MEDICAL: PathMNIST (CC BY 4.0) — colon pathology, 9 tissue classes")
print("=" * 100)
d = vl_data.prepare_vqa("pathmnist", max_train=2000)
if d is None:
    print("  LOADER RETURNED NONE")
    raise SystemExit(1)
m = d["meta"]
print(f"  source      : {m['source']}")
print(f"  classes ({len(m['names'])}): {m['names']}")
print(f"  train {d['Xtr'].shape}  test {d['Xte'].shape}  question len {m['qlen']}")
print(f"  pixels in [{d['Xtr'].min():.3f}, {d['Xtr'].max():.3f}]")
print(f"  answer vocab ({len(d['avocab'])}): {d['avocab']}")
print()
print("  is the label distribution sane?")
u, c = np.unique(d["ytr"], return_counts=True)
for i, n in zip(u, c):
    print(f"    class {i} {m['names'][int(i)]:<28} {n:>6}  ({n/len(d['ytr'])*100:5.1f}%)")
print()
print("  a sample question/answer pair:")
for i in range(3):
    q = " ".join(d["qvocab"][t] for t in d["Qtr"][i] if t != 0)
    print(f"    Q {q:<24} A {d['avocab'][d['Atr'][i,0]]:<24} (true class {m['names'][int(d['ytr'][i])]})")
print()
print("  IMPORTANT: the 9 classes are NOT balanced in PathMNIST, so the majority-class")
print("  baseline is not 1/9. The trainer prints the real majority baseline.")
print("=" * 100)
PY
