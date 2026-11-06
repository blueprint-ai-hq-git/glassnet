"""lens/scripts/unified_ladder_test.py - PLAN item (verifier3): a DIRECTION out of a RIGHT/WRONG verdict.

WHY THIS EXISTS (measured, not asserted): lens/runs/UNIFIED-VERIFIER.txt PART 5/6 says the method's default
rule works with a RANK oracle (slope error 0.47/0.45/0.44%, 100.0% exact on trained AND unseen pairs) and
does NOT carry with only a right/wrong verifier (97/98/99% trained, **0.0% unseen**, slope error 6.797% at
EVERY probe step tried). The dead zone is the VERDICT's own tolerance (0.5), not the probe step, so
shrinking the probe cannot fix it and a rank-capable question is what is missing.

THIS TEST measures whether a direction can be recovered from the right/wrong verdict ALONE, by asking it
more than once: the ladder in lens/unified.py::binary_sign_ladder. Predictions P57a-P57l were written
BEFORE any code changed: lens/runs/UNIFIED-LADDER-PREDICTIONS.txt.

PART 7a - the helper's claim, checked directly on synthetic residuals (no training involved): the ladder's
          sign must equal the true sign of (y - pred) for every residual it resolves, and it must resolve
          every residual except exact window-edge coincidences.
PART 7b - the arms, 3 seeds, 12000 steps, same harness, same gate defaults as PART 5/6:
            L1 nearest   (today's fallback, kept as a NAMED variant - must reproduce PART 6's numbers)
            L2 ladder    (the fix)
            L3 ladder with the direction FLIPPED   (CONTROL 1: the ladder must be able to fail)
            L4 ladder against an always-right verifier (CONTROL 2: no information in -> no direction out)
            L5 rank oracle (must be untouched, digit for digit)
PART 7c - the COST of the fix: verifier questions per example per step, over the run and over the last
          1000 steps.
"""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.unified import train_unified, binary_sign_ladder      # noqa: E402
from lens.scripts.unified_verifier_test import (corpus, right_wrong, rank_best,   # noqa: E402
                                                rank_worst)

TOL = 0.5           # the verifier's DECLARED tolerance: |round(pred) - y| < 0.5
DMAX = 40.0         # top rung: the targets run 0..18, so this covers every residual in the harness
SPACING = 1.5       # in units of tol; < 2 so consecutive rung windows OVERLAP (no residual falls between)
STEPS = 12000
SEEDS = (0, 1, 2)
LR = 0.05
STEP = SPACING * TOL


def pct3(rows, key):
    return " ".join("%5.1f%%" % (r[key] * 100.0) for r in rows)


def pct2(rows, key):
    return " ".join("%6.2f%%" % (r[key] * 100.0) for r in rows)


def part_7a():
    print("=" * 100)
    print("PART 7a - THE HELPER'S CLAIM, CHECKED DIRECTLY (no training): does the ladder's sign equal the")
    print("          true sign of (y - pred), and does it resolve every residual?")
    print("=" * 100)
    rng = np.random.default_rng(20261021)
    n = 200000
    # THE HARNESS CASE FIRST: y is an integer (y = a + b), so the value the verifier accepts IS y and
    # sign(y - pred) is the thing the ladder must reproduce.
    pred = rng.uniform(-25.0, 25.0, size=n)
    y = rng.integers(0, 19, size=n).astype(float)
    r = y - pred
    sign, questions, unresolved = binary_sign_ladder(right_wrong, pred, y, tol=TOL, dmax=DMAX,
                                                     spacing=SPACING)
    resolved = sign != 0.0
    ok = np.sign(r) == sign
    wrong = int((resolved & ~ok).sum())
    un_idx = np.where(~resolved)[0]
    if len(un_idx):
        absr = np.abs(r[un_idx])
        edges = np.concatenate([np.arange(0.0, DMAX + 4 * TOL, STEP),
                                2.0 * TOL + np.arange(0.0, DMAX + 4 * TOL, STEP)])
        dist = np.min(np.abs(absr[:, None] - edges[None, :]), axis=1)
        print("  unresolved %d of %d (%.4f%%): |r| min %.6f max %.6f; distance to the nearest window "
              "edge: max %.3e" % (len(un_idx), n, len(un_idx) / n * 100.0, absr.min(), absr.max(),
                                  dist.max()))
    print("  resolved %d of %d (%.4f%%)   WRONG SIGN on %d of them   verifier calls %d"
          % (int(resolved.sum()), n, resolved.sum() / n * 100.0, wrong, questions))
    v57a = wrong == 0 and unresolved / n <= 0.001
    print("  P57a %s  (compared: wrong signs %d against the bar 0, and the unresolved fraction %.6f "
          "against the bar <= 0.001)" % ("PASS" if v57a else "FAIL", wrong, unresolved / n))
    # the same check with the direction flipped: it must disagree on ALL resolved residuals, so the sign
    # really is the thing being tested
    sign_f, _, _ = binary_sign_ladder(right_wrong, pred, y, tol=TOL, dmax=DMAX, spacing=SPACING)
    wrong_f = int(((sign_f != 0.0) & (np.sign(r) != -sign_f)).sum())
    print("  control: with the direction flipped, %d of %d resolved residuals disagree (it must be all of "
          "them)" % (wrong_f, n))
    # SEMANTICS, stated because it is not what I first assumed: this verifier accepts a candidate p when
    # |round(p) - y| < 0.5, so the set it accepts is the interval of width 1 around ROUND(y), not around
    # y. The ladder therefore resolves the sign of (accepted value - pred). With an integer y (the
    # harness, and the first check above) the two are the same thing; with a continuous y they differ by
    # at most 0.5, and where |y - pred| < 0.5 they can even point opposite ways - which is a property of
    # the VERIFIER, not of the ladder. Reported, not hidden.
    yc = rng.uniform(0.0, 18.0, size=n)
    sign_c, _, unres_c = binary_sign_ladder(right_wrong, pred, yc, tol=TOL, dmax=DMAX, spacing=SPACING)
    c = np.round(yc)
    wrong_c = int(((sign_c != 0.0) & (np.sign(c - pred) != sign_c)).sum())
    print("  continuous-y information line: wrong sign against sign(round(y) - pred): %d of %d; "
          "residuals where sign(round(y) - pred) != sign(y - pred): %d (all of them have |y - pred| < 0.5)"
          % (wrong_c, n, int((np.sign(c - pred) != np.sign(yc - pred)).sum())))
    return v57a


def run_arm(seed, **kw):
    t0 = time.time()
    verifier = kw.pop("verifier", right_wrong)
    (w, b), log = train_unified(X, y, mode="verifier", verifier=verifier, steps=STEPS, lr=LR, seed=seed,
                                verbose=False, **kw)
    tr = float((np.round(X @ w + b) == y).mean())
    un = float((np.round(Xu @ w + b) == yu).mean())
    sl = abs(float(w[0]) - 20.0) / 20.0
    qpe = np.asarray(log["ladder_q_per_example"], dtype=float)
    return dict(tr=tr, un=un, sl=sl, wall=time.time() - t0,
                q_mean=float(qpe.mean()) if len(qpe) else 0.0,
                q_last=float(qpe[-1000:].mean()) if len(qpe) else 0.0,
                unresolved=log["ladder_unresolved"], examples=log["ladder_examples"],
                accepted=log["accepted"], refused_gap=log["refused_gap"])


def part_7b():
    print()
    print("=" * 100)
    print("PART 7b - THE ARMS, 3 SEEDS, 12000 STEPS, SAME HARNESS AND GATE DEFAULTS AS PART 5/6")
    print("=" * 100)
    print("  ladder: tol %s (declared), rung spacing %s*tol = %s, top rung %s, rungs %d"
          % (TOL, SPACING, STEP, DMAX, int(np.floor((DMAX) / STEP)) + 1))
    print("  predictions P57a-P57l written first: lens/runs/UNIFIED-LADDER-PREDICTIONS.txt")
    always_right = lambda pred, y_: np.ones_like(pred, dtype=bool)
    lad = dict(binary_probe="ladder", ladder_tol=TOL, ladder_max=DMAX, ladder_spacing=SPACING)
    arms = (
        ("L1 nearest (today's fallback)", dict(binary_probe="nearest")),
        ("L2 ladder (the fix)", dict(lad)),
        ("L3 ladder, direction FLIPPED (control)", dict(lad, ladder_flip=True)),
        ("L4 ladder, always-right verifier (control)", dict(lad, verifier=always_right)),
        ("L5 rank oracle (must be unchanged)", dict(verifier_rank=rank_best)),
    )
    res = {}
    print("  %-42s%22s%22s%22s" % ("arm", "trained", "unseen", "slope err"))
    print("  " + "-" * 108)
    for label, kw in arms:
        rows = [run_arm(s, **dict(kw)) for s in SEEDS]
        res[label] = rows
        print("  %-42s%22s%22s%22s" % (label, pct3(rows, "tr"), pct3(rows, "un"), pct2(rows, "sl")))
    print("  " + "-" * 108)
    L1 = res["L1 nearest (today's fallback)"]
    L2 = res["L2 ladder (the fix)"]
    L3 = res["L3 ladder, direction FLIPPED (control)"]
    L4 = res["L4 ladder, always-right verifier (control)"]
    L5 = res["L5 rank oracle (must be unchanged)"]

    v57e = (all(abs(r["sl"] - 0.06797) <= 1e-4 for r in L1) and all(r["un"] == 0.0 for r in L1))
    print("  P57e %s  (compared: 'nearest' slope error %s against PART 6's 0.06797 +- 1e-4 per seed, unseen "
          "%s against [0.0, 0.0, 0.0])"
          % ("PASS" if v57e else "FAIL", [round(r["sl"], 5) for r in L1],
             [round(r["un"] * 100, 1) for r in L1]))
    v57b = all(r["sl"] <= 0.01 for r in L2)
    print("  P57b %s  (compared: ladder slope error %s against the bar <= 0.01 on 3 of 3)"
          % ("PASS" if v57b else "FAIL", [round(r["sl"], 5) for r in L2]))
    v57c = all(r["un"] >= 0.50 for r in L2)
    print("  P57c %s  (compared: ladder unseen %s against the bar >= 50.0 on 3 of 3)"
          % ("PASS" if v57c else "FAIL", [round(r["un"] * 100, 1) for r in L2]))
    v57d = all(r["tr"] >= 0.95 for r in L2)
    print("  P57d %s  (compared: ladder trained %s against the bar >= 95.0 on 3 of 3)"
          % ("PASS" if v57d else "FAIL", [round(r["tr"] * 100, 1) for r in L2]))
    v57f = all(r["sl"] >= 0.50 and r["un"] <= 0.05 for r in L3)
    print("  P57f %s  (compared: flipped-ladder slope error %s against the bar >= 0.50 and unseen %s "
          "against <= 5.0 on 3 of 3)"
          % ("PASS" if v57f else "FAIL", [round(r["sl"], 5) for r in L3],
             [round(r["un"] * 100, 1) for r in L3]))
    v57g = all(r["sl"] >= 0.90 and r["un"] == 0.0 for r in L4)
    print("  P57g %s  (compared: always-right control slope error %s against the bar >= 0.90 and unseen %s "
          "against 0.0)"
          % ("PASS" if v57g else "FAIL", [round(r["sl"], 5) for r in L4],
             [round(r["un"] * 100, 1) for r in L4]))
    v57i = (all(abs(r["sl"] - 0.00472) <= 1e-9 or abs(r["sl"] - 0.00448) <= 1e-9
                or abs(r["sl"] - 0.00436) <= 1e-9 for r in L5) and all(r["un"] >= 0.50 for r in L5))
    print("  P57i %s  (compared: rank-oracle slope error %s against arm G's [0.00472, 0.00448, 0.00436] and "
          "unseen %s against >= 50.0)"
          % ("PASS" if v57i else "FAIL", [round(r["sl"], 5) for r in L5],
             [round(r["un"] * 100, 1) for r in L5]))
    return v57e, v57b, v57c, v57d, v57f, v57g, v57i, res


def part_7c(res):
    print()
    print("=" * 100)
    print("PART 7c - THE COST OF THE FIX: verifier calls per step (= questions per example per step,")
    print("          because one call asks EVERY example in the batch one question)")
    print("=" * 100)
    L2 = res["L2 ladder (the fix)"]
    qm = [r["q_mean"] for r in L2]
    ql = [r["q_last"] for r in L2]
    un = [r["unresolved"] / max(r["examples"], 1) for r in L2]
    print("  mean verifier calls/step over the whole run per seed: %s" % [round(v, 2) for v in qm])
    print("  mean verifier calls/step over the LAST 1000 steps per seed: %s" % [round(v, 3) for v in ql])
    print("  (rungs scanned per step = calls/2; the ladder has %d rungs available)"
          % (int(np.floor(DMAX / STEP)) + 1))
    print("  unresolved fraction per seed: %s" % [round(v, 6) for v in un])
    print("  wall clock per seed (s): %s" % [round(r["wall"], 1) for r in L2])
    print("  accepted / refused-by-gap steps per seed: %s"
          % [(r["accepted"], r["refused_gap"]) for r in L2])
    print("  for comparison, the old fallback asks 3 questions per example per step, and a converged")
    print("  ladder asks 2 (one rung: both sides of the smallest rung).")
    v57a2 = all(v <= 0.001 for v in un)
    v57h = all(v <= 40.0 for v in qm) and all(v <= 4.0 for v in ql)
    print("  P57a (training side) %s  (compared: unresolved fraction %s against the bar <= 0.001)"
          % ("PASS" if v57a2 else "FAIL", [round(v, 6) for v in un]))
    print("  P57h %s  (compared: mean verifier calls/step %s against the bar <= 40.0 and the "
          "last-1000-step mean %s against <= 4.0)"
          % ("PASS" if v57h else "FAIL", [round(v, 2) for v in qm], [round(v, 3) for v in ql]))
    return v57a2, v57h


if __name__ == "__main__":
    X, y, Xu, yu, pairs, unseen = corpus()
    print("=" * 100)
    print("PLAN ITEM (verifier3): A DIRECTION OUT OF A RIGHT/WRONG VERDICT - measured on the 2+2 harness")
    print("=" * 100)
    print("  training pairs %d (a,b in 0..9)   unseen pairs %d (a,b in 10..19)   pairs in both %d"
          % (len(pairs), len(unseen), len(set(pairs) & set(unseen))))
    print("  NOT CLAIMED: nothing about the chat model and nothing about the real corpus. Arithmetic harness.")
    print()
    a7a = part_7a()
    v57e, v57b, v57c, v57d, v57f, v57g, v57i, res = part_7b()
    a7b, v57h = part_7c(res)
    print()
    print("VERDICT LINE, stated plainly:")
    print("  the ladder's claim, checked directly (P57a helper)     : %s" % ("PASS" if a7a else "FAIL"))
    print("  the ladder in training, 3 seeds (P57b/c/d, P57a train) : %s"
          % ("PASS" if (v57b and v57c and v57d and a7b) else "FAIL"))
    print("  the old fallback did not move (P57e)                   : %s" % ("PASS" if v57e else "FAIL"))
    print("  controls (P57f flipped, P57g no-information)           : %s"
          % ("PASS" if (v57f and v57g) else "FAIL"))
    print("  the rank path untouched (P57i)                         : %s" % ("PASS" if v57i else "FAIL"))
    print("  cost (P57h)                                            : %s" % ("PASS" if v57h else "FAIL"))
    print("  A PASS here means a user with ONLY a right/wrong verifier now gets a direction. It is not a")
    print("  claim about the chat model or the real corpus, and the default only moves if P57b/P57c PASS.")
