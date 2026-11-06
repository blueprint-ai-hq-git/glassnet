"""val_window_leak_check.py - how much of the held-out split is actually held out?

WHY THIS EXISTS
---------------
The gate test of 2026-10-21 (PLAN item d2b, lens/runs/GATE-TEST.txt) measured something nobody was looking
for: on data/chat/prepared_clean, 5 of 256 probe windows drawn from the HELD-OUT split also appear in the
training set, matched on CONTENT (the 32 window tokens plus the target token), not on indices. A hard rule of
this project is that no figure may be quoted from a val that overlaps train, so the size of that overlap has
to be a number before the val numbers are used again.

WHAT IT MEASURES, AND HOW
  * the VAL side is ALWAYS complete: every window the trainer could draw from the held-out split.
  * the TRAIN side is the windows the model ACTUALLY trains on: the same call the trainer makes
    (chat_data.make_windows(..., which=0, max_windows=N, seed=S)). With --full-train it is every train window
    instead, which is an upper bound; that mode is only usable on a small corpus.
  * a val window counts as leaked if its exact content is present on the train side.
  The window builder is the trainer's OWN function - this script never reimplements it, because a check that
  rebuilds the data path tests a different data path.

Run:
  python3 lens/scripts/val_window_leak_check.py data/chat/prepared_clean 32 1500000 2
  python3 lens/scripts/val_window_leak_check.py data/chat/prepared_clean 32 full 0
"""
import hashlib
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(PROJ)


def hset(X, Y):
    return {hashlib.md5(np.ascontiguousarray(X[i]).tobytes() + int(Y[i]).to_bytes(4, "little")).digest()
            for i in range(len(X))}


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/chat/prepared_clean"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    cap = sys.argv[3] if len(sys.argv) > 3 else "1500000"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    os.environ["CHAT_DATA_DIR"] = os.path.abspath(corpus)
    sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
    import chat_data                                            # noqa: E402  (env must be set first)

    d = chat_data.build()
    ids, bounds, split = d["ids"], d["bounds"], d["split"]
    maxw = None if cap == "full" else int(cap)
    Xtr, Ytr = chat_data.make_windows(ids, bounds, split, window, which=0, max_windows=maxw, seed=seed)
    Xva, Yva = chat_data.make_windows(ids, bounds, split, window, which=1)
    print(f"{corpus}: window {window}, train side {len(Xtr):,} windows "
          f"({'EVERY train window' if maxw is None else f'sampled with seed {seed}'}), "
          f"val side {len(Xva):,} windows (complete)")

    th = hset(Xtr, Ytr)
    vh = hset(Xva, Yva)
    overlap = len(th & vh)
    frac = overlap / max(len(Xva), 1)
    print(f"  val windows whose exact content is present on the train side: {overlap:,} of {len(Xva):,} "
          f"({frac * 100:.3f}%)")
    print(f"  duplicate contents WITHIN the train side: {len(Xtr) - len(th):,} "
          f"(how repetitive the corpus is at this window size)")
    print(f"  duplicate contents WITHIN the val side: {len(Xva) - len(vh):,}")
    res = {"corpus": corpus, "window": window, "train_side": ("full" if maxw is None else int(maxw)),
           "train_seed": seed, "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
           "val_windows_in_train": int(overlap), "val_window_overlap_fraction": float(frac),
           "duplicate_contents_within_train": int(len(Xtr) - len(th)),
           "duplicate_contents_within_val": int(len(Xva) - len(vh))}
    p = os.path.join(PROJ, "lens", "runs", "val_window_leak_check.json")
    old = json.load(open(p)) if os.path.exists(p) else []
    old = [r for r in old if not (r.get("corpus") == corpus and r.get("train_side") == res["train_side"])]
    json.dump(old + [res], open(p, "w"), indent=1)
    print(f"  written to {p}")


if __name__ == "__main__":
    main()
