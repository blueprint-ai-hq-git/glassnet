"""aggregate_ktest.py — read the 9 ktest logs and decide k1/k2 with printed numbers beside verdicts.

(The verdict-check rule: every compared value prints NEXT TO its PASS/FAIL — three verdict bugs in
this project were right numbers with wrong conclusions.)
"""
import json
import os

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(P, "lens", "runs", "image")


def main():
    accs = {"none": [], "real": [], "dummy": []}
    passrate = {"none": [], "real": [], "dummy": []}
    for arm in accs:
        for seed in (0, 1, 2):
            f = os.path.join(R, f"ktest_{arm}_s{seed}_log.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            a = d["results"]["readable"]["history"][-1]["test_acc"] * 100
            accs[arm].append(a)
            passrate[arm].append(d["audit"]["n_pass"])
    print("=" * 92)
    print("SEE-ONLY K-TRANSFER (test acc %, 10k TEST images per run — one use each, logged)")
    print("=" * 92)
    means = {}
    for arm in accs:
        v = np.asarray(accs[arm])
        if len(v) == 0:
            print(f"  {arm:<6} NO ARTIFACTS YET")
            continue
        means[arm] = (float(v.mean()), float(v.std()))
        print(f"  {arm:<6} K={'12' if arm=='none' else '22'}  test {v.mean():6.2f} ± {v.std():.2f}"
              f"   seeds {[round(x,2) for x in v]}   audit pass {sorted(passrate[arm])}")
    if len(means) == 3:
        dn, rn, dm = means["none"], means["real"], means["dummy"]
        k1 = (rn[0] - dn[0]) > 2.0
        k2 = abs(rn[0] - dm[0]) <= 2.0
        print()
        print(f"  k1 real-none = {rn[0]:.2f} - {dn[0]:.2f} = {rn[0]-dn[0]:+.2f} > 2.0 -> "
              f"{'PASS' if k1 else 'FAIL'}")
        print(f"  k2 |real-dummy| = {abs(rn[0]-dm[0]):.2f} <= 2.0 -> {'PASS' if k2 else 'FAIL'}")
        noise = rn[1] + dm[1] + dn[1]
        print(f"  (combined seed noise ~±{noise:.2f} — differences inside it are 'no difference')")
        if k1 and k2:
            v = "K-GAIN TRANSFERS to see-only; and it is WIDTH again (real~dummy): same story as the " \
                "seetalk sweep, now across both model families."
        elif not k1 and not k2:
            v = "K-GAIN DOES NOT TRANSFER: the +4.4 was caption-position-specific, not concept-count."
        else:
            v = "MIXED: read the numbers above; do not narrate around them."
        print(f"  VERDICT: {v}")
        json.dump({"means": means, "gates": {"k1": bool(k1), "k2": bool(k2)}, "verdict": v},
                  open(os.path.join(R, "ktest_verdict.json"), "w"), indent=1)
    else:
        print("  (incomplete run — verdict withheld, not guessed)")


if __name__ == "__main__":
    main()
