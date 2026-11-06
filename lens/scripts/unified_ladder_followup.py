"""lens/scripts/unified_ladder_followup.py - the follow-up to PART 7: three bars were WORDED WRONG, and the
default moves because P57b/P57c PASSED.

THE ORIGINAL VERDICTS STAND. Nothing here retunes them; each FAIL stays in lens/runs/UNIFIED-LADDER.txt and
is repeated here with the comparison that was actually intended, labelled CORRECTED, exactly as this project
handled the earlier gate disagreement (PLAN.txt, 2026-10-20 14:5x: "the FAIL stands; the old line is kept,
corrected").

  P57e FAIL as worded: the bar applied PART 6's ONE-SEED stored number (0.06797) to THREE seeds. PART 6 ran
       seed 0 only, so seeds 1 and 2 (0.06489, 0.05948) were never measured by it and cannot match.
       CORRECTED CHECK: seed 0 against the stored number.
  P57i FAIL as worded: the bar compared FULL-PRECISION floats against the FIVE-DECIMAL numbers arm G prints,
       with a 1e-9 tolerance - my error, and the printed values are identical.
       CORRECTED CHECK: round to 5 decimals, which is the precision arm G's stored numbers have.
  P57a FAIL as worded: the direct check swept residuals out to +-43, past the ladder's top rung (dmax 40.0),
       so it counted by design the residuals the ladder never asks about. CORRECTED CHECK: split the
       unresolved count by whether |r| is inside the ladder's range (dmax + tol = 40.5).

AND the default move (P57l), because P57b and P57c PASSED: binary_probe defaults to 'ladder' now; 'nearest'
stays a named variant so every stored number that describes it is still reproducible. This script checks
that the smoke test's DATA arm, TEACHER arm and STAGE 2 are digit-for-digit identical before and after the
move, so the only thing the move changed is the verifier arm.
"""
import os
import sys

import numpy as np

_ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, _ROOT)

from lens.unified import train_unified, binary_sign_ladder               # noqa: E402
from lens.scripts.unified_verifier_test import corpus, right_wrong, rank_best   # noqa: E402

TOL, DMAX, SPACING, STEPS, LR = 0.5, 40.0, 1.5, 12000, 0.05
ARM_G = [0.00472, 0.00448, 0.00436]


def slope(w):
    return abs(float(w[0]) - 20.0) / 20.0


def main():
    X, y, Xu, yu, pairs, unseen = corpus()
    print()
    print("=" * 100)
    print("PART 8 - THE CORRECTED CHECKS (the original FAILs stand) AND THE DEFAULT MOVE")
    print("=" * 100)

    # ---- P57e CORRECTED: seed 0 against PART 6's stored number
    (w1, b1), _ = train_unified(X, y, mode="verifier", verifier=right_wrong, steps=STEPS, lr=LR, seed=0,
                                verbose=False, binary_probe="nearest")
    s1 = slope(w1)
    ok_e = abs(s1 - 0.06797) <= 1e-4
    print("  P57e CORRECTED %s  (compared: 'nearest' seed 0 slope error %.5f against PART 6's stored 0.06797 "
          "+- 1e-4)" % ("PASS" if ok_e else "FAIL", s1))

    # ---- P57i CORRECTED: five decimals, the precision arm G's stored numbers have
    sls = []
    for seed in (0, 1, 2):
        (w5, b5), _ = train_unified(X, y, mode="verifier", verifier=right_wrong, steps=STEPS, lr=LR,
                                    seed=seed, verbose=False, verifier_rank=rank_best)
        sls.append(slope(w5))
    r5 = [round(v, 5) for v in sls]
    ok_i = r5 == ARM_G
    print("  P57i CORRECTED %s  (compared: rank-oracle slope error at 5 decimals %s against arm G's stored "
          "%s)" % ("PASS" if ok_i else "FAIL", r5, ARM_G))

    # ---- P57a CORRECTED: the unresolved count split by the ladder's own range
    rng = np.random.default_rng(20261021)
    n = 200000
    pred = rng.uniform(-25.0, 25.0, size=n)
    yv = rng.integers(0, 19, size=n).astype(float)
    r = yv - pred
    sign, calls, unresolved = binary_sign_ladder(right_wrong, pred, yv, tol=TOL, dmax=DMAX, spacing=SPACING)
    inside = np.abs(r) <= DMAX + TOL
    un_in = int((~inside & (sign == 0.0)).sum())
    un_out = int((inside & (sign == 0.0)).sum())
    wrong_in = int((inside & (sign != 0.0) & (np.sign(r) != sign)).sum())
    print("  P57a CORRECTED  (compared, decomposed instead of pooled): residuals with |r| <= dmax + tol = "
          "%.1f: %d of %d, unresolved %d, wrong sign %d | residuals beyond the top rung: %d, unresolved %d "
          "(by design - the ladder is not asked about them)"
          % (DMAX + TOL, int(inside.sum()), n, un_out, wrong_in, int((~inside).sum()), un_in))
    ok_a = un_out == 0 and wrong_in == 0
    print("  P57a CORRECTED %s  (compared: unresolved inside the range %d against the bar 0, wrong signs "
          "inside the range %d against the bar 0)" % ("PASS" if ok_a else "FAIL", un_out, wrong_in))

    # ---- THE DEFAULT MOVE: the smoke test before vs after, split by arm
    pre = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/UNIFIED-RUN-pre-ladder.txt"
    new = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/UNIFIED-RUN.txt"
    if not (os.path.exists(pre) and os.path.exists(new)):
        print("  P57l split check NOT RUN: %s or %s is missing" % (pre, new))
        return
    lp = open(pre).read().splitlines()
    ln = open(new).read().splitlines()

    def slice_out(lines):
        i_data = next(i for i, l in enumerate(lines) if l.strip().startswith("mode 'data':"))
        i_ver = next(i for i, l in enumerate(lines) if l.strip().startswith("mode 'verifier':"))
        i_st2 = next(i for i, l in enumerate(lines) if l.strip().startswith("STAGE 2 -"))
        return lines[i_data:i_ver], lines[i_st2:]

    ap, bp = slice_out(lp)
    an, bn = slice_out(ln)
    same_a = ap == an
    same_b = bp == bn
    print("  P57j AFTER THE MOVE %s  (compared: the data+teacher arm block %d lines against %d, identical: "
          "%s; the STAGE 2 block %d lines against %d, identical: %s)"
          % ("PASS" if (same_a and same_b) else "FAIL", len(ap), len(an), same_a, len(bp), len(bn), same_b))
    if not (same_a and same_b):
        for i, (x, z) in enumerate(zip(ap, an)):
            if x != z:
                print("    data/teacher line %d: %r -> %r" % (i, x, z))
        for i, (x, z) in enumerate(zip(bp, bn)):
            if x != z:
                print("    STAGE 2 line %d: %r -> %r" % (i, x, z))
    # and the verifier arm MUST have moved - otherwise the move did nothing
    def ver_block(lines):
        i_ver = next(i for i, l in enumerate(lines) if l.strip().startswith("mode 'verifier':"))
        i_st2 = next(i for i, l in enumerate(lines) if l.strip().startswith("STAGE 2 -"))
        return lines[i_ver:i_st2]
    vp, vn = ver_block(lp), ver_block(ln)
    print("  the verifier arm moved: %s" % (vp != vn))
    for x, z in zip(vp, vn):
        if x != z:
            print("    %s\n -> %s" % (x, z))


if __name__ == "__main__":
    main()
