"""DOES PRUNING 14 CONCEPTS TO 6 COST ANY ACCURACY?

The pruning measurement says 6 of 14 concepts predict all 14 at average R^2 >= 0.90, so a person must read 6 names instead of 14.
That is only a win if it costs NOTHING that matters. So: train the same linear classifier on all 14 concepts and on the 6, and
compare held-out accuracy on the real Fashion-MNIST classes.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 the 6-concept classifier is within 1.0 accuracy point of the 14-concept one.
  P2 neither collapses (both beat the most-frequent-class baseline).
FALSIFIER: if the 6 lose more than 1.0 point, the redundancy is real but not free, and the honest count of concepts a person must
  read goes back up.
"""
import gzip
import json
import os
import struct
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", os.environ.get("GLASSNET_ROOT", "."))
KEEP = ["ink_amount", "ink_entropy", "unnamed_13", "ink_cols_nonzero", "corner_ink", "unnamed_11"]


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def fit_predict(Xtr, ytr, Xte, k=10, ridge=1e-6):
    """Linear classifier by exact least squares on one-hot targets. Our own linear algebra."""
    n = len(Xtr)
    A = np.column_stack([Xtr, np.ones(n)])
    Y = np.eye(k, dtype=np.float64)[ytr]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = np.column_stack([Xte, np.ones(len(Xte))]) @ coef
    return pred.argmax(axis=1)


def main():
    z = np.load(os.path.join(ROOT, "data", "concepts.npz"))
    Cte = z["Cte"].astype(np.float64)
    lo, hi = z["lo"].astype(np.float64), z["hi"].astype(np.float64)
    span = np.where((hi - lo) > 1e-8, hi - lo, 1.0)
    Yc = (Cte - lo) / span
    names = list(json.load(open(os.path.join(ROOT, "data", "concepts_disp10.json")))["names"])
    while len(names) < Cte.shape[1]:
        names.append("unnamed_%d" % len(names))
    y = read_idx(os.path.join(ROOT, "data", "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    n = min(len(Yc), len(y))
    Yc, y = Yc[:n], y[:n]

    nh = n // 2
    base = np.bincount(y[nh:]).max() / len(y[nh:])
    print("=== does pruning 14 concepts to 6 cost accuracy? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("samples %d (fit %d, test %d) | most-frequent-class baseline %.2f%%" % (n, nh, n - nh, 100 * base))

    idx14 = list(range(Cte.shape[1]))
    idx6 = [names.index(k) for k in KEEP]
    for label, idx in (("all 14 concepts", idx14), ("the 6 concepts", idx6)):
        pred = fit_predict(Yc[:nh][:, idx], y[:nh], Yc[nh:][:, idx])
        acc = 100.0 * (pred == y[nh:]).mean()
        print("  %-16s -> held-out accuracy %.2f%%" % (label, acc))
    p14 = fit_predict(Yc[:nh][:, idx14], y[:nh], Yc[nh:][:, idx14])
    p6 = fit_predict(Yc[:nh][:, idx6], y[:nh], Yc[nh:][:, idx6])
    a14 = 100.0 * (p14 == y[nh:]).mean()
    a6 = 100.0 * (p6 == y[nh:]).mean()
    print()
    print("RESULT 14 concepts %.2f%% vs 6 concepts %.2f%% | difference %+.2f points" % (a14, a6, a6 - a14))
    print("P1 within 1.0 point: %s" % ("MET" if abs(a6 - a14) <= 1.0 else "NOT MET"))
    print("P2 both beat the baseline: %s" % ("MET" if min(a6, a14) > 100 * base else "NOT MET"))
    if abs(a6 - a14) > 1.0:
        print("FALSIFIER FIRED: pruning is not free, the count a person must read goes back up.")
        print("OUTCOME: PRUNING_COSTS_ACCURACY")
    else:
        print("OUTCOME: PRUNING_IS_FREE")
    print("CONCEPT-PRUNE-COST COMPLETE")


if __name__ == "__main__":
    main()
