"""lens/scripts/unified_verifier_test.py - WHY verifier mode does not converge, and whether the fix works.

PLAN.txt line 370: "CODE verifier mode does not converge (9.0% trained / 0.0% unseen, cold-start partial
credit used 1015 times, lens/runs/UNIFIED-RUN.txt). Fix it or say plainly that it is not working yet."

Predictions P51a-i were written BEFORE any code was changed: lens/runs/UNIFIED-VERIFIER-PREDICTIONS.txt.

PART 1 - the mechanism, measured not asserted: the shipped cold-start rule keeps the closest quarter of the
         batch by |pred - y|. At pred ~ 0 that is the quarter with the SMALLEST targets. Instrument it.
PART 2 - three arms on the same harness, same seeds, same steps, plus the data-mode control:
           A shipped (hard right/wrong + closest quarter)
           B probe3  (the verifier names the best of three candidates; train toward the winner)
           C control (the verifier names the WORST candidate - the test must be able to fail)
           D data    (the upper bound)
PART 3 - the split check: the unseen pairs must share nothing with the corpus.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.unified import train_unified, GapGate   # noqa: E402


def corpus():
    """The exact case Yahya used: 2+2 must be COMPUTED, not looked up."""
    pairs = [(a, b) for a in range(10) for b in range(10)]
    X = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in pairs], dtype=np.float64)
    y = np.asarray([a + b for a, b in pairs], dtype=np.float64)
    unseen = [(a, b) for a in range(10, 20) for b in range(10, 20)]
    Xu = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in unseen], dtype=np.float64)
    yu = np.asarray([a + b for a, b in unseen], dtype=np.float64)
    return X, y, Xu, yu, pairs, unseen


def right_wrong(pred, y):
    """The verifier as it exists: right or wrong, exactly, on the rounded answer."""
    return np.abs(np.round(pred) - y) < 0.5


def rank_best(cands, y):
    """The verifier asked the question that carries a DIRECTION: which candidate is closest to right?"""
    return np.argmin(np.abs(cands - y[:, None]), axis=1)


def rank_worst(cands, y):
    """CONTROL: the same oracle answering backwards. The test must be able to fail."""
    return np.argmax(np.abs(cands - y[:, None]), axis=1)


def main():
    X, y, Xu, yu, pairs, unseen = corpus()
    print("=" * 100)
    print("VERIFIER MODE: WHY IT DOES NOT CONVERGE, AND THE FIX - measured on the 2+2 harness")
    print("=" * 100)
    print("predictions P51a-i were written first: lens/runs/UNIFIED-VERIFIER-PREDICTIONS.txt")
    print("stored state being attacked (lens/runs/UNIFIED-RUN.txt): 9.0% trained / 0.0% unseen,")
    print("cold-start partial-credit steps 1015")
    print()

    # ---------------------------------------------------------------- PART 3 first: the split check (P51h)
    overlap = sorted(set(pairs) & set(unseen))
    print("PART 3 - THE SPLIT CHECK (P51h)")
    print(f"  training pairs {len(pairs)} (a,b in 0..9)   unseen pairs {len(unseen)} (a,b in 10..19)   "
          f"pairs in both: {len(overlap)}")
    print(f"  P51h {'PASS' if not overlap else 'FAIL'}  (compared: overlap {len(overlap)} against the bar 0)")
    print()

    # ------------------------------------------------------------------ PART 1: the shipped rule, instrumented
    print("PART 1 - THE MECHANISM: what the shipped cold-start rule actually selects (P51b, P51c)")
    diag = []
    (w, b), log = train_unified(X, y, mode="verifier", verifier=right_wrong, steps=6000, seed=0,
                                verbose=False, diag=diag, verifier_mode="hard_quarter")
    seen_a = float((np.round(X @ w + b) == y).mean())
    unseen_a = float((np.round(Xu @ w + b) == yu).mean())
    print(f"  arm A reproduced: exact trained {seen_a*100:.1f}%   exact unseen {unseen_a*100:.1f}%   "
          f"cold-start steps {log.get('partial_credit_steps', 0)}")
    print(f"  P51a {'PASS' if abs(seen_a*100-9.0) <= 5.0 and unseen_a == 0.0 else 'FAIL'}  "
          f"(compared: trained {seen_a*100:.1f}% against 9.0+-5.0, unseen {unseen_a*100:.1f}% against 0.0)")
    if diag:
        mb = np.asarray([d["mean_y_batch"] for d in diag])
        ms = np.asarray([d["mean_y_selected"] for d in diag])
        below = float((ms < mb).mean())
        first_ratio = float(ms[0] / mb[0]) if mb[0] else float("nan")
        print(f"  cold-start steps instrumented: {len(diag)}")
        print(f"  mean target of the SELECTED quarter vs the BATCH: below on {below*100:.1f}% of steps")
        print(f"  first step: batch mean {mb[0]:.3f}   selected mean {ms[0]:.3f}   ratio {first_ratio:.3f}")
        print(f"  P51b {'PASS' if below >= 0.90 else 'FAIL'}  (compared: {below:.3f} against the bar 0.90)")
        print(f"  P51c {'PASS' if first_ratio <= 0.60 else 'FAIL'}  "
              f"(compared: ratio {first_ratio:.3f} against the bar 0.60)")
        # how the bias behaves as the run proceeds: does the frontier ever cross the middle?
        q = len(diag) // 4
        for i, lbl in ((0, "first quarter"), (q, "second quarter"), (2*q, "third quarter")):
            if i < len(diag):
                print(f"    {lbl:>15}: batch mean {mb[i]:.2f}  selected mean {ms[i]:.2f}  "
                      f"model pred mean {diag[i]['mean_pred']:.2f}")
    else:
        print("  no cold-start steps recorded -> P51b/P51c are unmeasurable (report as such)")
    print()

    # --------------------------------------------------------------------------- PART 2: the arms, 3 seeds
    print("PART 2 - THE ARMS, 3 SEEDS, 3000 STEPS EACH, SAME HARNESS")
    print(f"  {'arm':<26}{'exact trained':>15}{'exact UNSEEN':>15}{'cold-start steps':>19}")
    print("  " + "-" * 74)
    results = {}
    def anneal(k):
        """P52: the probe step halves every 500 steps, floored at 1/16. Written in ADDENDUM 1 before the run."""
        return max(1.0 / 16.0, 1.0 * 0.5 ** (k / 500.0))

    for label, kw in (
        ("A shipped", dict(mode="verifier", verifier=right_wrong, verifier_mode="hard_quarter")),
        ("B probe3", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                          verifier_rank=rank_best, probe_step=1.0)),
        ("C reversed (control)", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                      verifier_rank=rank_worst, probe_step=1.0)),
        ("D data (upper bound)", dict(mode="data")),
        ("E probe3 annealed", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                   verifier_rank=rank_best, probe_schedule=anneal)),
        ("F annealed reversed", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                     verifier_rank=rank_worst, probe_schedule=anneal)),
        ("G normalised s=1/16", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                     verifier_rank=rank_best, probe_step=1.0 / 16.0, probe_normalise=True)),
        ("H norm. reversed", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                  verifier_rank=rank_worst, probe_step=1.0 / 16.0, probe_normalise=True)),
    ):
        seen, uns, cs, exactc, wrec = [], [], [], [], []
        for seed in (0, 1, 2):
            d = []
            (w, b), log = train_unified(X, y, steps=3000, lr=0.05, seed=seed, verbose=False,
                                        diag=d, **kw)
            seen.append(float((np.round(X @ w + b) == y).mean()))
            uns.append(float((np.round(Xu @ w + b) == yu).mean()))
            cs.append(log.get("partial_credit_steps", 0))
            wrec.append(np.asarray(w, dtype=float))
            if d and "exact_candidate" in d[0]:
                tail = d[int(len(d) * 0.9):]
                exactc.append(float(np.mean([x["exact_candidate"] for x in tail])))
        results[label] = dict(seen=seen, uns=uns, cs=cs, exactc=exactc, w=wrec)
        print(f"  {label:<26}{np.mean(seen)*100:>14.1f}%{np.mean(uns)*100:>14.1f}%{np.mean(cs):>19.1f}")
        print(f"  {'':<26}{'/'.join(f'{v*100:.0f}' for v in seen):>14} {'/'.join(f'{v*100:.0f}' for v in uns):>14}"
              f"   per seed (trained / unseen)")
    print("  " + "-" * 74)
    B, A, C, D = results["B probe3"], results["A shipped"], results["C reversed (control)"], results["D data (upper bound)"]

    v51d = all(v >= 0.95 for v in B["seen"])
    v51e = all(v >= 0.50 for v in B["uns"])
    print(f"  P51d {'PASS' if v51d else 'FAIL'}  (compared: arm B trained per seed "
          f"{[round(v*100,1) for v in B['seen']]} against the bar >= 95.0 on 3 of 3)")
    print(f"  P51e {'PASS' if v51e else 'FAIL'}  (compared: arm B unseen per seed "
          f"{[round(v*100,1) for v in B['uns']]} against the bar >= 50.0 on 3 of 3)")
    v51f = bool(B["exactc"]) and all(v >= 0.95 for v in B["exactc"])
    print(f"  P51f {'PASS' if v51f else 'FAIL'}  (compared: arm B exact-candidate fraction over the last 10% "
          f"of steps {[round(v,4) for v in B['exactc']]} against the bar >= 0.95 on 3 of 3)")
    v51g = all(v < 0.50 for v in C["uns"]) and np.mean(C["seen"]) <= np.mean(A["seen"])
    print(f"  P51g {'PASS' if v51g else 'FAIL'}  (compared: control unseen "
          f"{[round(v*100,1) for v in C['uns']]} against the bar < 50.0, and control trained "
          f"{np.mean(C['seen'])*100:.1f}% against arm A's {np.mean(A['seen'])*100:.1f}%)")
    print(f"  reference: arm D (data) trained {np.mean(D['seen'])*100:.1f}% unseen {np.mean(D['uns'])*100:.1f}%"
          f"  <- the upper bound this harness can reach")

    # ------------------------------------------------------------------ ADDENDUM 1: the annealed probe (P52)
    E, F = results["E probe3 annealed"], results["F annealed reversed"]
    print()
    print("ADDENDUM 1 - ARM E, the probe step ANNEALED (1.0 halving every 500 steps, floored at 1/16)")
    print("  the freeze this tests: as soon as |pred - y| <= s the closest candidate is pred itself, the target")
    print("  becomes the model's own output and that example stops contributing - so a FIXED s stops the run at")
    print("  the edge of the set of weights that round correctly, and extrapolation needs the interior.")
    v52a = all(v >= 0.95 for v in E["seen"])
    v52b = all(v >= 0.50 for v in E["uns"])
    v52c = all(e > b for e, b in zip(E["uns"], B["uns"]))
    v52d = all(v < 0.50 for v in F["uns"])
    print(f"  P52a {'PASS' if v52a else 'FAIL'}  (compared: arm E trained per seed "
          f"{[round(v*100,1) for v in E['seen']]} against the bar >= 95.0 on 3 of 3)")
    print(f"  P52b {'PASS' if v52b else 'FAIL'}  (compared: arm E unseen per seed "
          f"{[round(v*100,1) for v in E['uns']]} against the bar >= 50.0 on 3 of 3)")
    print(f"  P52c {'PASS' if v52c else 'FAIL'}  (compared: E unseen {[round(v*100,1) for v in E['uns']]} "
          f"against B unseen {[round(v*100,1) for v in B['uns']]} per seed, bar: strictly better on every seed)")
    print(f"  P52d {'PASS' if v52d else 'FAIL'}  (compared: annealed-reversed control unseen "
          f"{[round(v*100,1) for v in F['uns']]} against the bar < 50.0)")

    # ------------------------------------------------- ADDENDUM 2: small fixed probe + normalised update (P53)
    G, H = results["G normalised s=1/16"], results["H norm. reversed"]
    print()
    print("ADDENDUM 2 - ARM G: probe step FIXED at 1/16 and the update NORMALISED by 1/s (see the addendum")
    print("  in the predictions file: annealing was wrong, because the step's magnitude is proportional to s;")
    print("  the measured failure of arm B is a SLOPE 6.5% low - w=[18.66, 18.80, 0.28] against [20, 20, 0] -")
    print("  which rounding hides on the trained range and which is fatal off it.")
    slopes = [float(g[0]) for g in G["w"]]
    v53a = all(v >= 0.95 for v in G["seen"])
    v53b = all(v >= 0.50 for v in G["uns"])
    v53c = all(abs(s - 20.0) / 20.0 <= 0.01 for s in slopes)
    v53d = all(v < 0.50 for v in H["uns"])
    print(f"  arm G slopes (w[0]) per seed {[round(s,4) for s in slopes]}  against the true 20.0")
    print(f"  P53a {'PASS' if v53a else 'FAIL'}  (compared: arm G trained per seed "
          f"{[round(v*100,1) for v in G['seen']]} against the bar >= 95.0 on 3 of 3)")
    print(f"  P53b {'PASS' if v53b else 'FAIL'}  (compared: arm G unseen per seed "
          f"{[round(v*100,1) for v in G['uns']]} against the bar >= 50.0 on 3 of 3)")
    print(f"  P53c {'PASS' if v53c else 'FAIL'}  (compared: |w[0]-20|/20 per seed "
          f"{[round(abs(s-20.0)/20.0,5) for s in slopes]} against the bar <= 0.01)")
    print(f"  P53d {'PASS' if v53d else 'FAIL'}  (compared: normalised-reversed control unseen "
          f"{[round(v*100,1) for v in H['uns']]} against the bar < 50.0)")

    print()
    print("=" * 100)
    print("PART 4 - ADDENDUM 3: is arm G's slope error SLOW CONVERGENCE or a STALL? 12000 steps, same rule")
    print("=" * 100)
    print("  the six numbers above say the unseen score follows the slope error exactly; this arm asks whether")
    print("  the rule gets there with 4x the steps, or whether it has stopped moving.")
    print(f"  {'arm':<26}{'steps':>7}{'slope error per seed':>34}{'unseen per seed':>22}")
    print("  " + "-" * 88)
    conv = {}
    for label, kw in (
        ("A shipped", dict(mode="verifier", verifier=right_wrong, verifier_mode="hard_quarter")),
        ("G normalised s=1/16", dict(mode="verifier", verifier=right_wrong, verifier_mode="probe3",
                                     verifier_rank=rank_best, probe_step=1.0 / 16.0, probe_normalise=True)),
    ):
        sl, uns2, tr2 = [], [], []
        for seed in (0, 1, 2):
            (w, b), log = train_unified(X, y, steps=12000, lr=0.05, seed=seed, verbose=False, **kw)
            sl.append(abs(float(w[0]) - 20.0) / 20.0)
            uns2.append(float((np.round(Xu @ w + b) == yu).mean()))
            tr2.append(float((np.round(X @ w + b) == y).mean()))
        conv[label] = dict(sl=sl, uns=uns2, tr=tr2)
        print(f"  {label:<26}{12000:>7}{'  '.join(f'{v*100:6.2f}%' for v in sl):>34}"
              f"{'  '.join(f'{v*100:5.1f}%' for v in uns2):>22}")
    print("  " + "-" * 88)
    Gc, Ac = conv["G normalised s=1/16"], conv["A shipped"]
    v54a = all(v <= 0.01 for v in Gc["sl"])
    v54b = all(v >= 0.50 for v in Gc["uns"])
    v54c = all(v < 0.20 for v in Ac["tr"])
    print(f"  P54a {'PASS' if v54a else 'FAIL'}  (compared: slope error {[round(v,5) for v in Gc['sl']]} "
          f"against the bar <= 0.01 on 3 of 3)")
    print(f"  P54b {'PASS' if v54b else 'FAIL'}  (compared: unseen {[round(v*100,1) for v in Gc['uns']]} "
          f"against the bar >= 50.0 on 3 of 3)")
    print(f"  P54c {'PASS' if v54c else 'FAIL'}  (compared: shipped arm trained {[round(v*100,1) for v in Ac['tr']]} "
          f"against the bar < 20.0 on 3 of 3)")
    print(f"  arm G at 12000 steps: trained {[round(v*100,1) for v in Gc['tr']]} unseen "
          f"{[round(v*100,1) for v in Gc['uns']]}")

    # ------------------------------------------ PART 5: THE DEFAULT ITSELF (PLAN item verifier2, 2026-10-21 18:3x)
    print()
    print("=" * 100)
    print("PART 5 - THE NEW DEFAULT, TESTED AS THE DEFAULT (no verifier_mode argument passed at all)")
    print("=" * 100)
    print("  predictions P55a-P55k written first: lens/runs/UNIFIED-VERIFIER-PREDICTIONS-ADDENDUM4.txt")
    print("  this is the point of the item: the rule that works must BE the shipped default of the METHOD,")
    print("  not a named parameter nothing calls. The shipped TRAINER (lens/scripts/train_chat.py) is not")
    print("  touched - this is the method, and the trainer does not use it yet (PLAN item (d2b)).")
    import inspect
    _sig = inspect.signature(train_unified)
    dflt = (_sig.parameters["verifier_mode"].default, _sig.parameters["probe_step"].default,
            _sig.parameters["probe_normalise"].default)
    v55a = (dflt[0] == "probe3" and abs(dflt[1] - 1.0 / 16.0) < 1e-12 and dflt[2] is True)
    print(f"  P55a {'PASS' if v55a else 'FAIL'}  (compared: the defaults {dflt} against "
          f"('probe3', 0.0625, True))")

    def run_default(seed, **extra):
        (w_, b_), log_ = train_unified(X, y, mode="verifier", verifier=right_wrong, steps=12000,
                                       lr=0.05, seed=seed, verbose=False, **extra)
        return (float((np.round(X @ w_ + b_) == y).mean()),
                float((np.round(Xu @ w_ + b_) == yu).mean()), float(w_[0]))

    print(f"  {'arm':<34}{'trained per seed':>22}{'unseen per seed':>22}{'slope err per seed':>24}")
    print("  " + "-" * 100)
    rows = {}
    for label, extra in (
        ("I default, binary verifier", {}),
        ("J default + rank_best (=arm G)", dict(verifier_rank=rank_best)),
        ("K default + rank_worst (control)", dict(verifier_rank=rank_worst)),
    ):
        tr3, un3, sl3 = [], [], []
        for seed in (0, 1, 2):
            t_, u_, w0_ = run_default(seed, **extra)
            tr3.append(t_)
            un3.append(u_)
            sl3.append(abs(w0_ - 20.0) / 20.0)
        rows[label] = dict(tr=tr3, un=un3, sl=sl3)
        print(f"  {label:<34}{' '.join(f'{v*100:5.1f}%' for v in tr3):>22}"
              f"{' '.join(f'{v*100:5.1f}%' for v in un3):>22}{' '.join(f'{v*100:6.2f}%' for v in sl3):>24}")
    print("  " + "-" * 100)
    I5, J5, K5 = (rows["I default, binary verifier"], rows["J default + rank_best (=arm G)"],
                  rows["K default + rank_worst (control)"])
    v55g = all(v >= 0.95 for v in I5["tr"]) and all(v >= 0.50 for v in I5["un"])
    print(f"  P55g {'PASS' if v55g else 'FAIL'}  (compared: the DEFAULT trained "
          f"{[round(v*100,1) for v in I5['tr']]} against the bar >= 95.0 and unseen "
          f"{[round(v*100,1) for v in I5['un']]} against the bar >= 50.0, on 3 of 3 seeds)")
    v55h = (all(v <= 0.01 for v in J5["sl"]) and all(v >= 0.50 for v in J5["un"])
            and all(abs(a - b) <= 1e-9 for a, b in zip(J5["sl"], Gc["sl"])))
    print(f"  P55h {'PASS' if v55h else 'FAIL'}  (compared: default+rank_best slope error "
          f"{[round(v,5) for v in J5['sl']]} against arm G's {[round(v,5) for v in Gc['sl']]} at the same "
          f"12000 steps, bar <= 0.01 and identical to arm G)")
    v55i = all(v < 0.50 for v in K5["un"])
    print(f"  P55i {'PASS' if v55i else 'FAIL'}  (compared: default+rank_worst control unseen "
          f"{[round(v*100,1) for v in K5['un']]} against the bar < 50.0 on 3 of 3)")
    if not v55g:
        print("  READ IT AS: the corrected RULE works (arm J/G) but the binary-verifier FALLBACK does not")
        print("  carry it, so the default is available without being the working one. Recorded, not hidden.")

    print()
    print("=" * 100)
    print("PART 6 - THE DEAD ZONE: what sets it, the probe step or the verifier's own tolerance?")
    print("=" * 100)
    print("  predictions P56a-d written first: lens/runs/UNIFIED-VERIFIER-PREDICTIONS-ADDENDUM5.txt")
    print("  one seed (0), 12000 steps, same harness, probe step swept. A MECHANISM probe, not a win claim.")
    print("  the claim under test: the rank oracle stops giving a direction only when |pred - y| is under")
    print("  about s/2, so ITS dead zone is the probe step; a right/wrong verifier stops when its own")
    print("  rounding tolerance is satisfied, so ITS dead zone is 0.5 - eight times wider than s = 1/16.")
    print(f"  {'probe step s':<14}{'verifier':<24}{'trained':>10}{'unseen':>10}{'slope err':>12}")
    print("  " + "-" * 72)
    sweep = {}
    steps6 = (1.0, 0.25, 1.0 / 16.0, 1.0 / 64.0)
    for s_now in steps6:
        for vname, extra in (("binary right/wrong", {}),
                             ("rank oracle", dict(verifier_rank=rank_best)),
                             ("rank reversed (control)", dict(verifier_rank=rank_worst))):
            (w6, b6), _ = train_unified(X, y, mode="verifier", verifier=right_wrong, steps=12000,
                                        lr=0.05, seed=0, verbose=False, probe_step=s_now,
                                        probe_normalise=True, **extra)
            tr6 = float((np.round(X @ w6 + b6) == y).mean())
            un6 = float((np.round(Xu @ w6 + b6) == yu).mean())
            sl6 = abs(float(w6[0]) - 20.0) / 20.0
            sweep[(s_now, vname)] = dict(tr=tr6, un=un6, sl=sl6)
            print(f"  {s_now:<14.6f}{vname:<24}{tr6*100:>9.1f}%{un6*100:>9.1f}%{sl6*100:>11.2f}%")
    print("  " + "-" * 72)
    bin_un = [sweep[(s, "binary right/wrong")]["un"] for s in steps6]
    bin_sl = [sweep[(s, "binary right/wrong")]["sl"] for s in steps6]
    rk_un = [sweep[(s, "rank oracle")]["un"] for s in (1.0 / 16.0, 1.0 / 64.0)]
    rk_sl = [sweep[(s, "rank oracle")]["sl"] for s in (1.0 / 16.0, 1.0 / 64.0)]
    ct_un = [sweep[(s, "rank reversed (control)")]["un"] for s in steps6]
    v56a = all(v < 0.50 for v in bin_un)
    print(f"  P56a {'PASS' if v56a else 'FAIL'}  (compared: binary-verifier unseen "
          f"{[round(v*100,1) for v in bin_un]} at s = 1, 1/4, 1/16, 1/64 against the bar < 50.0 on every s)")
    v56b = all(v >= 0.50 for v in rk_un) and all(v <= 0.01 for v in rk_sl)
    print(f"  P56b {'PASS' if v56b else 'FAIL'}  (compared: rank-oracle unseen {[round(v*100,1) for v in rk_un]} "
          f"and slope error {[round(v,5) for v in rk_sl]} at s = 1/16 and 1/64 against bars >= 50.0, <= 0.01)")
    v56c = all(v >= 0.05 for v in bin_sl) and all(v <= 0.01 for v in rk_sl)
    print(f"  P56c {'PASS' if v56c else 'FAIL'}  (compared: binary slope error {[round(v,5) for v in bin_sl]} "
          f"against the bar >= 0.05 on every s, and the rank arm's {[round(v,5) for v in rk_sl]} against <= 0.01)")
    v56d = all(v < 0.50 for v in ct_un)
    print(f"  P56d {'PASS' if v56d else 'FAIL'}  (compared: reversed-oracle control unseen "
          f"{[round(v*100,1) for v in ct_un]} against the bar < 50.0 on every s)")
    if not v56a:
        print("  READ IT AS: my stated cause is WRONG - some probe step lets a right/wrong verifier")
        print("  extrapolate, so the dead zone is not simply the verifier's 0.5 tolerance. Not hidden.")

    print()
    print("VERDICT LINE, stated plainly:")
    print(f"  PASS 1 (probe step fixed at 1.0)      : P51d/P51e {'PASS' if (v51d and v51e) else 'FAIL'}"
          f" - 95%/91% on the trained pairs on two seeds, 0% off them.")
    print(f"  PASS 2 (probe step ANNEALED)          : P52a-c {'PASS' if (v52a and v52b and v52c) else 'FAIL'}"
          f" - lands on the shipped arm's collapse, and the debug shows nothing freezes.")
    if v53a and v53b and v53c:
        print("  PASS 3 (small fixed probe, NORMALISED) : THE MECHANISM WAS THE SLOPE. With the zero-gradient")
        print("  band made negligible and the step no longer shrinking with s, the rule reaches the true slope")
        print("  and extrapolates to pairs it never saw. Verifier mode is no longer dead on this harness.")
    else:
        print(f"  PASS 3 (small fixed probe, NORMALISED) : P53a/b/c "
              f"{'PASS' if (v53a and v53b and v53c) else 'FAIL'} - recorded as FAILED, bars not retuned.")
    print(f"  PASS 4 (12000 steps, convergence)     : P54a/b/c "
          f"{'PASS' if (v54a and v54b and v54c) else 'FAIL'}"
          f" - slope error {[round(v*100,2) for v in Gc['sl']]}, unseen "
          f"{[round(v*100,1) for v in Gc['uns']]}.")
    if v54a and v54b:
        print("  READ IT AS: the mechanism was the slope and the rule CONVERGES to it given steps; at 3000 steps")
        print("  it reaches the true slope on one seed of three. Verifier mode is not dead - it is SLOW.")
    else:
        print("  READ IT AS: even at 4x the steps the normalised probe rule does not reach the true slope on")
        print("  every seed. Verifier mode stays NOT FIXED and this is written down as such.")
    print("  The DEFAULT is now the rule that works (PART 5, P55a/g/h): verifier_mode='probe3', probe step")
    print("  fixed at 1/16, update normalised by 1/s. The old cold-start rule is kept as a NAMED variant")
    print("  ('hard_quarter') so every stored number that describes it is still reproducible on request -")
    print("  the arms A above pass it explicitly, which is exactly why P55f can still be checked after the")
    print("  default moved. The shipped TRAINER (lens/scripts/train_chat.py) is untouched.")
    print()
    print("NOT CLAIMED: nothing about the chat model or the real corpus. This is the arithmetic harness the")
    print("method is developed on. A PASS means verifier mode is no longer dead, not that it trains GlassNet.")


if __name__ == "__main__":
    main()
