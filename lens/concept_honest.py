"""DOES A CONCEPT NEURON MEAN WHAT ITS NAME SAYS?  with the instrument certified first.

WHY THIS EXISTS (2026-11-22): the recorded state of the readable middle layer is that, in the chat model, it answers by
CANCELLATION - of 36 concepts only 17 supply their own label and 19 OPPOSE it, and the top-8 supply 68.7% against an 80% bar.
Before I try to fix that, I have to certify the instrument that produced the 17/19 split, because this file's own history
(recorded in lens/concepts.py) already lists three occasions where an AUDIT was the thing that was broken:

  1. an audit compared a concept set the model had not been trained on;
  2. an audit measured the wrong statistic and reported 0/9 on a model that was fine;
  3. a best-perplexity checkpoint turned out to be an epoch-1 model.

So this audit is built to be falsifiable BY CONSTRUCTION: it plants four kinds of neuron whose correct verdict is known in
advance, and it REFUSES to report a verdict on the real concepts unless it classifies every planted one correctly.

THE AUDIT, and why it is not just a correlation:
  Concept labels are correlated with each other (brightness and upper-left ink move together), so a neuron that tracks
  "how bright is it" will look like it tracks several names at once. A neuron can only be said to MEAN its own name if it
  carries information about its label that the OTHER labels do not already carry. So for neuron j:

      own          = |corr(neuron_j, label_j)|
      redundant    = max over other labels k of |corr(label_k, label_j)|   <- how well the rest explain label_j
      unique       = own - redundant

      HONEST if own >= 0.50 and unique > 0.10
      and the reported "what it actually tracks" is argmax over ALL labels of |corr(neuron_j, label_k)|.

THE FOUR PLANTED NEURONS (verdict known before the audit runs):
  HONEST_COPY    = label_0 itself                      -> must be called HONEST
  IMPOSTOR       = label_1 wearing the name of label_0 -> must be called DISHONEST (this is the "secretly detects
                   something else" case the project brief demands to see caught)
  MIXED          = 0.6*label_0 + 0.4*label_1            -> must NOT be called HONEST (it is two things at once)
  NOISE          = random numbers                       -> must be called DISHONEST

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 the audit classifies all four planted neurons correctly.
  P2 the audit's verdict on the real trained neurons is reported only if P1 holds; otherwise it prints
     INSTRUMENT_INVALID and refuses.
FALSIFIER: if the IMPOSTOR is called HONEST, the instrument is broken and NO verdict on the real concepts may be quoted.
"""
import argparse
import gzip
import json
import os
import struct
import sys
import time

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", os.environ.get("GLASSNET_ROOT", "."))

OWN_BAR = 0.50
# THE BAR THAT FAILED ITS OWN TEST (2026-11-22 13:56): a fixed "unique > 0.10" bar called a PERFECT copy of label 0
# DISHONEST, because that label is 92% predictable from the other concepts, so nothing can be 0.10 more informative than
# them. The ceiling for any label is (1 - its redundancy), so the criterion must be a FRACTION of the label's own
# predictable information, not an absolute margin. A neuron is honest when it captures over half of what makes its label
# unique, and when its strongest match is its own name.
UNIQUE_FRAC_BAR = 0.50


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_images(n):
    """Fashion-MNIST test images, flattened and scaled to 0..1."""
    p = os.path.join(ROOT, "data", "t10k-images-idx3-ubyte.gz")
    if not os.path.exists(p):
        raise SystemExit("missing %s - cannot audit against real concepts without the images" % p)
    img = read_idx(p)[:n].reshape(-1, 784).astype(np.float32) / 255.0
    return img


def fit_concept_layer(X, Y, ridge=1e-3):
    """EXACT least squares, solved by hand with our own linear algebra - no library optimiser, no autograd.

    WHY NOT GRADIENT DESCENT: the first version trained with plain GD, and the audit then reported own=0.00 for EVERY
    real concept neuron - the layer had not learned anything in the step budget, so the verdict measured my learning rate
    rather than the concepts. A closed-form fit removes the learning rate as a knob entirely, so the audit measures the
    ceiling of what a LINEAR concept neuron can capture, not how well I tuned the training."""
    n, d = X.shape
    k = Y.shape[1]
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + ridge * n * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xc.T @ Yc).astype(np.float32)
    b = (ym - xm @ W).astype(np.float32)
    mse = float(((X @ W + b - Y) ** 2).mean())
    print("  exact least-squares fit: mse %.5f on the fitting half" % mse)
    return W, b


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0


def audit_neuron(neuron, labels, j):
    """The verdict for one neuron, with the evidence that produced it."""
    k = labels.shape[1]
    cors = np.array([corr(neuron, labels[:, i]) for i in range(k)])
    own = abs(cors[j])
    others = [abs(corr(labels[:, i], labels[:, j])) for i in range(k) if i != j]
    redundant = max(others) if others else 0.0
    unique = own - redundant
    ceiling = max(1.0 - redundant, 1e-6)          # the most any neuron could capture beyond the other labels
    unique_frac = unique / ceiling
    tracks = int(np.argmax(np.abs(cors)))
    # it must (a) match its OWN name best, (b) track it strongly, (c) capture over half of what makes the label unique
    honest = (tracks == j) and (own >= OWN_BAR) and (unique_frac > UNIQUE_FRAC_BAR)
    return dict(own=own, redundant=redundant, unique=unique, unique_frac=unique_frac, ceiling=ceiling,
                tracks=tracks, honest=bool(honest), cors=cors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="images used (test split)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    names = json.load(open(os.path.join(ROOT, "data", "concepts_disp10.json")))["names"]
    z = np.load(os.path.join(ROOT, "data", "concepts.npz"))
    Cte, lo, hi = z["Cte"], z["lo"], z["hi"]
    k = Cte.shape[1]
    # the display file names 10 of the 14 label columns; the rest get an honest placeholder so nothing is mislabelled
    names = list(names)[:k]
    while len(names) < k:
        names.append("unnamed_%d" % len(names))

    X = load_images(a.n)
    n = min(len(X), len(Cte))
    X, Y = X[:n], Cte[:n]
    # standardise the labels with the SAME lo/hi the concepts file ships, so a "correlation" means something
    span = np.where((hi - lo) > 1e-8, hi - lo, 1.0)
    Yn = ((Y - lo) / span).astype(np.float32)

    print("=== do the concept neurons mean their names? %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("images %d, concept labels %d: %s" % (n, k, ", ".join(names)))
    print()
    # FIT AND AUDIT ON DIFFERENT HALVES, so a concept neuron cannot be called honest merely by memorising the images
    nh = n // 2
    Xf, Yf = X[:nh], Yn[:nh]
    Xa, Ya = X[nh:], Yn[nh:]
    print("--- fitting the concept decoders (exact least squares, hand written) on %d images ---" % nh)
    W, b = fit_concept_layer(Xf, Yf)
    neur = Xa @ W + b           # (na, k) activations of the concept neurons, on images NEVER used for the fit
    print("    auditing on %d held-out images" % len(Xa))

    # ---------------- CERTIFY THE INSTRUMENT FIRST ----------------
    print()
    print("--- INSTRUMENT TEST: four planted neurons, verdict known before the audit runs ---")
    rng = np.random.default_rng(999)
    planted = []
    na = len(Ya)
    hon = Ya[:, 0] + 0.02 * rng.standard_normal(na)
    imp = Ya[:, 1] + 0.02 * rng.standard_normal(na)
    mix = 0.6 * Ya[:, 0] + 0.4 * Ya[:, 1] + 0.02 * rng.standard_normal(na)
    noi = rng.standard_normal(na)
    cases = [("HONEST_COPY (is label 0, named label 0)", hon, 0, "HONEST"),
             ("IMPOSTOR (is label 1, named label 0)", imp, 0, "DISHONEST"),
             ("MIXED (0.6 label0 + 0.4 label1)", mix, 0, "NOT_HONEST"),
             ("NOISE (random)", noi, 0, "DISHONEST")]
    ok_all = True
    Yn_audit = Ya
    for label, vec, j, expected in cases:
        r = audit_neuron(vec, Yn_audit, j)
        got = "HONEST" if r["honest"] else "DISHONEST"
        if expected == "NOT_HONEST":
            good = not r["honest"]
        else:
            good = (got == expected)
        ok_all &= good
        print("  %-46s own=%.2f redundant=%.2f frac=%.2f -> %-9s (tracks %-16s) %s"
              % (label, r["own"], r["redundant"], r["unique_frac"], got,
                 names[r["tracks"]] if r["tracks"] < len(names) else r["tracks"],
                 "CORRECT" if good else "*** WRONG ***"))
        planted.append(good)
    print()
    print("P1 the audit classifies all four planted neurons correctly: %s"
          % ("MET" if ok_all else "NOT MET"))
    if not ok_all:
        print("FALSIFIER FIRED: the impostor case is the one that matters - if a neuron that secretly tracks another")
        print("concept can be called HONEST, the instrument is broken.")
        print("OUTCOME: INSTRUMENT_INVALID - no verdict on the real concepts is quoted.")
        print("CONCEPT-AUDIT COMPLETE")
        return

    # ---------------- now the real neurons ----------------
    print()
    print("INSTRUMENT VALID (all four planted cases correct). Now the real trained neurons:")
    print()
    print("  %-22s %6s %10s %6s  %-22s %s" % ("named concept", "own", "redund", "frac", "actually tracks", "verdict"))
    honest_count = 0
    for j in range(k):
        r = audit_neuron(neur[:, j], Ya, j)
        honest_count += r["honest"]
        print("  %-22s %6.2f %10.2f %6.2f  %-22s %s"
              % (names[j], r["own"], r["redundant"], r["unique_frac"], names[r["tracks"]],
                 "HONEST" if r["honest"] else ("WRONG NAME" if r["tracks"] != j else "weak")))
    print()
    print("RESULT %d of %d concept neurons mean their own name (matches its own label best, own >= %.2f, and captures"
          " more than %.0f%% of what makes its label unique)" % (honest_count, k, OWN_BAR, 100 * UNIQUE_FRAC_BAR))
    print("CONCEPT-AUDIT COMPLETE  (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
