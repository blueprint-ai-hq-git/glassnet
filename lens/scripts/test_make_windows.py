#!/usr/bin/env python3
"""test_make_windows.py - the rewritten window builder must reproduce the old one EXACTLY.

The old algorithm (kept here as a reference implementation) built every window in a Python loop; the new one
is vectorised and can subsample without materialising the full set. A rewrite of the data path is exactly the
kind of change that silently shifts every recorded number, so it is checked against the old code, not against
expectations.
"""
import sys
import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, f"{ROOT}/lens/scripts")
import chat_data  # noqa: E402


def old_make_windows(ids, bounds, split, window, which, return_index=False):
    X, Y, S = [], [], []
    for j in range(len(bounds) - 1):
        if split[j] != which:
            continue
        a, b = bounds[j], bounds[j + 1]
        seg = ids[a:b]
        for s in range(0, len(seg) - window - 1, window // 2):
            X.append(seg[s:s + window]); Y.append(seg[s + window]); S.append(a + s)
    X = np.asarray(X, dtype=np.int64); Y = np.asarray(Y, dtype=np.int64)
    if return_index:
        return X, Y, np.asarray(S, dtype=np.int64)
    return X, Y


def main():
    rng = np.random.default_rng(0)
    ok = True
    for trial, (nconv, tlen) in enumerate([(6, 90), (12, 200), (3, 40), (25, 60)]):
        bounds = np.concatenate([[0], np.cumsum(rng.integers(tlen // 2, tlen, size=nconv))]).astype(np.int64)
        ids = rng.integers(0, 500, size=int(bounds[-1])).astype(np.int32)
        split = (rng.random(nconv) < 0.3).astype(np.int8)
        for which in (0, 1):
            for window in (8, 32):
                Xo, Yo, So = old_make_windows(ids, bounds, split, window, which, return_index=True)
                Xn, Yn, Sn = chat_data.make_windows(ids, bounds, split, window, which, return_index=True)
                if Xo.size == 0 and Xn.size == 0:
                    # both empty. The old code returned shape (0,) float64 because np.asarray([]) does; the
                    # new code returns (0, window) int64, which is the correct shape and dtype. Values agree.
                    same = True
                else:
                    same = (Xo.shape == Xn.shape and np.array_equal(Xo, Xn) and np.array_equal(Yo, Yn)
                            and np.array_equal(So, Sn))
                print(f"  trial {trial} which={which} window={window}: old {Xo.shape} new {Xn.shape} "
                      f"identical={same}")
                ok &= same
    # capped path: exactly N windows, and every row must be a REAL window of the corpus
    bounds = np.concatenate([[0], np.cumsum(np.full(40, 120))]).astype(np.int64)
    ids = np.random.default_rng(1).integers(0, 500, size=int(bounds[-1])).astype(np.int32)
    split = np.zeros(40, dtype=np.int8)
    # the corpus has 40 conversations x 6 windows = 240, so a cap of 100 must return exactly 100, and a cap
    # ABOVE the total must return the total (asking for 500 windows out of 240 cannot invent more)
    Xc, Yc, Sc = chat_data.make_windows(ids, bounds, split, 32, 0, return_index=True,
                                       max_windows=100, seed=7)
    Xall, _, _ = chat_data.make_windows(ids, bounds, split, 32, 0, return_index=True)
    print(f"  cap below total: requested 100 of {len(Xall)}, got {len(Xc)}")
    ok &= (len(Xc) == 100)
    Xbig, _, _ = chat_data.make_windows(ids, bounds, split, 32, 0, return_index=True,
                                       max_windows=5000, seed=7)
    print(f"  cap above total: requested 5000 of {len(Xall)}, got {len(Xbig)} (must equal the total)")
    ok &= (len(Xbig) == len(Xall))
    valid = bool(np.all(Yc == np.asarray(ids)[Sc + 32])) and bool(
        np.all(Xc[:, 0] == np.asarray(ids)[Sc]))
    print(f"  every capped row is a real window of the corpus: {valid}")
    ok &= valid
    print(f"  VERDICT: {'PASS' if ok else 'FAIL -- the rewrite does not match the old data path'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
