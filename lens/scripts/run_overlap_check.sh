#!/bin/bash
# IS THE RECORDED 63.85% CONTAMINATED? (corrected: the corpus path is read at IMPORT time, so the
# module must be RELOADED after changing CHAT_DATA_DIR -- the first version silently measured the same
# corpus twice and produced a meaningless 66.07% in both directions.)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import importlib
import os
import sys
import numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import chat_data
WIN = 32

def load(d):
    os.environ["CHAT_DATA_DIR"] = d
    importlib.reload(chat_data)              # <- CACHE is a module-level constant: reload, do not re-set
    x = chat_data.build()
    return x["ids"], np.asarray(x["bounds"]), np.asarray(x["split"]), x["vocab"]

def windows(d, which):
    ids, b, sp, vocab = load(d)
    X, Y = chat_data.make_windows(ids, b, sp, WIN, which=which)
    return X, Y, vocab

def void_rows(X):
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()

base = "/run/media/yahya/HARD DISK 4/GlassNet/data/chat/"
pairs = [("prepared_clean", "prepared_splitfix"), ("prepared_splitfix", "prepared_clean")]
for val_corpus, train_corpus in pairs:
    Xa, Ya, va = windows(base + val_corpus, 1)
    Xb, Yb, vb = windows(base + train_corpus, 0)
    same = (va == vb)
    if not same:
        print(f"  {val_corpus} vs {train_corpus}: vocabularies differ, overlap not measurable this way")
        continue
    ov = float(np.isin(void_rows(Xa), void_rows(Xb)).mean())
    print(f"  {val_corpus} val windows {len(Xa):,}  |  {train_corpus} train windows {len(Xb):,}  |  "
          f"vocab identical True")
    print(f"    share of {val_corpus} val windows appearing VERBATIM in {train_corpus} TRAIN text: "
          f"{ov*100:.2f}%")
PY