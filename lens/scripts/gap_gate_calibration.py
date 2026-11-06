"""lens/scripts/gap_gate_calibration.py - is the memorisation gate real or decoration?

WHY THIS EXISTS
  LEARN-NOT-MEMORISE.md section 2 says the model must REFUSE a step that memorises. lens/unified.py implements
  that as the gap gate. It was rewritten twice and is still not trusted: the first version refused 202 of 300
  honest steps (single noisy mini-batch read as memorisation), the EMA version refused 0 of 300 (too
  permissive). UNIFIED-TRAINING.md states the requirement exactly:

      "it must be tested on a corpus that CAN be memorised, where it has to fire, before it is trusted"

  This script does that test. Two corpora, chosen so the answer is known in advance:

    NOISE corpus (MEMORISATION IS THE ONLY OPTION)
      X ~ N(0,1), y ~ N(0,1) INDEPENDENT of X. Nothing generalises: any drop in training loss is the model
      fitting that particular batch. Held-out loss on fresh draws MUST rise. A correct gate refuses here.
      A gate that stays silent here is decoration, because it cannot detect the one thing it exists to detect.

    RULE corpus (LEARNING IS POSSIBLE)
      y = X @ w_true + 0.05*noise, w_true fixed. Held-out loss falls as the rule is found. The same gate
      MUST NOT block these steps. A gate that refuses here is an obstacle, not a rule.

PREDICTION, WRITTEN BEFORE RUNNING (2026-10-20 11:3x, from reading the code, not from a result)
  P1  The CURRENT gate ("ema_band": refuse when EMA_held > 1.05 * previous EMA_held) refuses < 10% of steps
      on the NOISE corpus. Reason: with beta=0.9 the candidate EMA is 0.9*old + 0.1*new, so the test
      cand > 1.05*old means 0.9*old + 0.1*new > 1.05*old, i.e. new_loss > 1.5 * old_EMA. A single step can
      almost never make the instantaneous held-out loss 50% worse than its own smoothed value, so the rule
      needs a 1.5x jump to fire. The smoothing and the tolerance multiply each other. I expect ~0 refusals.
  P2  A gate that compares the two smoothed series instead - "held-out EMA got worse AND train EMA got
      better" (the wording of LEARN-NOT-MEMORISE.md section 2a) - fires on a large fraction of NOISE steps
      and on few RULE steps.
  P3  Therefore verdict now: the gap gate is DECORATION as currently written. Verdict after the test must be
      read off the table below, and if P1 is right the current rule is marked FAIL, not "failed in spirit".

GATE VERDICT RULE, FIXED BEFORE THE RUN (binary)
  A configuration PASSES iff, for EVERY one of 3 seeds:
      refused fraction on NOISE >= 0.50   (it fires where memorisation is the only possible outcome)
      refused fraction on RULE  <= 0.20   (it does not block the steps that actually learn)
  Anything else is FAIL. Ties broken by the config that keeps RULE refusals lowest.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.unified import train_unified, GapGate  # the SHIPPED code, so the tested thing is the shipped thing


def make_corpus(kind, n=256, d=8, seed=1234):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, -1] = 1.0                      # bias column
    if kind == "noise":
        y = rng.normal(size=n)          # INDEPENDENT of X: memorisation is the only option
    elif kind == "rule":
        w_true = rng.normal(size=d)
        w_true[-1] = 0.0
        y = X @ w_true + 0.05 * rng.normal(size=n)
    else:
        raise ValueError(kind)
    return X, y


def run_one(kind, variant, beta, tol, seed, steps=300, lr=0.05, agreement_mode="one"):
    X, y = make_corpus(kind)
    gate = GapGate(beta=beta, tol=tol, variant=variant)
    (w, b), log = train_unified(X, y, mode="data", steps=steps, lr=lr, seed=seed,
                                gate=gate, agreement_mode=agreement_mode, verbose=False)
    checked = max(1, log["refused_gap"] + log["accepted"] + log["refused_agreement"])
    return {
        "kind": kind, "variant": variant, "beta": beta, "tol": tol, "seed": seed,
        "agreement_mode": agreement_mode,
        "refused_gap": log["refused_gap"],
        "refused_frac": log["refused_gap"] / checked,
        "accepted": log["accepted"],
        "heldout_first": log["heldout_first"],
        "heldout_last": log["heldout_last"],
    }


def finalist_effect(steps=300, lr=0.05):
    """PART 2 - the decisive test. Counting refusals is not enough: a gate is only real if REFUSING the steps
    it refuses actually PREVENTS the damage. So compare, on the same seeds, the held-out loss at the end with
    the gate ON versus the gate OFF. Nothing else changes.

    PREDICTION, WRITTEN BEFORE RUNNING (P4/P5):
      P4  On NOISE, the calibrated gate ends with a LOWER held-out loss than gate-off. Gate-off fits the
          batch and pays for it on fresh draws; a working gate refuses those steps. If the gate ends at the
          same held-out loss as gate-off, it is decoration no matter how many steps it refused.
      P5  On RULE (held-out loss ~8.6 and should fall), the calibrated gate ends within 20% of gate-off: it
          must not block the steps that learn.
    """
    print("\n" + "=" * 100)
    print("PART 2 - DOES REFUSING ACTUALLY PREVENT THE DAMAGE? (gate ON vs gate OFF, same seeds)")
    print("=" * 100)
    print("prediction P4: on NOISE the gate ends with a LOWER held-out loss than gate-off")
    print("prediction P5: on RULE the gate ends within 20% of gate-off (learning is not blocked)")
    print()
    configs = [("off", 0.9, 0.05), ("ema_band", 0.5, 0.0), ("ema_band", 0.9, 0.05),
               ("divergence", 0.5, 0.0)]
    out = {}
    print(f"{'corpus':<7}{'variant':<12}{'beta':>5}{'tol':>6} | {'held-out last':>18} {'train last':>18} "
          f"| {'refused':>9} {'accepted':>9}")
    print("-" * 100)
    for kind in ("noise", "rule"):
        for variant, beta, tol in configs:
            helds, trains, refs, accs = [], [], [], []
            for cseed in (1234, 7, 99):
                for tseed in (0, 1, 2):
                    X, y = make_corpus(kind, seed=cseed)
                    gate = GapGate(beta=beta, tol=tol, variant=variant)
                    (w, b), log = train_unified(X, y, mode="data", steps=steps, lr=lr, seed=tseed,
                                                gate=gate, verbose=False)
                    helds.append(log["heldout_last"])
                    trains.append(_train_last(X, y, w, b, gate_to_heldout_frac=0.25, seed=tseed))
                    refs.append(log["refused_gap"])
                    accs.append(log["accepted"])
            key = f"{kind}|{variant}|{beta}|{tol}"
            out[key] = {"heldout_last_mean": float(np.mean(helds)), "heldout_last_std": float(np.std(helds)),
                        "train_last_mean": float(np.mean(trains)),
                        "refused_mean": float(np.mean(refs)), "accepted_mean": float(np.mean(accs)),
                        "heldout_last_all": [float(v) for v in helds]}
            print(f"{kind:<7}{variant:<12}{beta:>5}{tol:>6} | "
                  f"{np.mean(helds):>10.4f}+-{np.std(helds):.4f} {np.mean(trains):>10.4f}+-{np.std(trains):.4f} "
                  f"| {np.mean(refs):>9.1f} {np.mean(accs):>9.1f}")
        print("-" * 100)

    # ---- the verdict, against the prediction, per corpus
    for kind in ("noise", "rule"):
        base = out[f"{kind}|off|0.9|0.05"]["heldout_last_mean"]
        cand = out[f"{kind}|ema_band|0.5|0.0"]["heldout_last_mean"]
        if kind == "noise":
            ok = cand < base
            print(f"NOISE  gate-off held-out {base:.4f}  vs calibrated gate {cand:.4f}  -> "
                  f"{'PASS - the gate prevents damage' if ok else 'FAIL - refusing changed nothing'}")
        else:
            ok = cand <= base * 1.20
            print(f"RULE   gate-off held-out {base:.4f}  vs calibrated gate {cand:.4f}  -> "
                  f"{'PASS - learning not blocked' if ok else 'FAIL - the gate blocks learning'}")
    return out


def _train_last(X, y, w, b, *, gate_to_heldout_frac, seed):
    """Training loss on the TRAIN split only (the held-out 25% excluded), so the number is comparable."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_held = max(8, int(len(X) * gate_to_heldout_frac))
    ti = order[n_held:]
    return float(np.mean((X[ti] @ w + b - y[ti]) ** 2))


def consistency_check(variant="ema_band", beta=0.5, tol=0.0):
    """PART 3 - LEARN-NOT-MEMORISE.md item 1: "same seed -> same number within tolerance, and different seeds
    -> the same verdict (this is the 'get that number consistent' requirement)". Nothing has ever checked it.

    PREDICTION P15, written before running: the same corpus and training seed reproduce the same refused count
    and the same final held-out EMA EXACTLY (the loop is pure numpy with a seeded generator), and 3 different
    seeds all give the same PASS/FAIL verdict against the rule fixed in this file."""
    print("\n" + "=" * 100)
    print("PART 3 - IS THE GATE NUMBER CONSISTENT? same seed twice, then 3 seeds' verdicts")
    print("=" * 100)
    print("prediction P15: same seed -> identical numbers; different seeds -> the same verdict")
    print()
    print("NOTE: every part below runs at the SHIPPED default agreement_mode='one' (step not throttled).")
    print("Part 1 was first measured with the old throttle; its numbers were re-taken here, and they changed.")
    runs = {}
    identical = True
    for kind in ("noise", "rule"):
        for seed in (0, 1, 2):
            a = run_one(kind, variant, beta, tol, seed)
            b = run_one(kind, variant, beta, tol, seed)
            same = (a["refused_gap"] == b["refused_gap"] and a["accepted"] == b["accepted"]
                    and a["heldout_last"] == b["heldout_last"])
            identical = identical and same
            runs[(kind, seed)] = a
            print(f"  {kind:<6} seed {seed}: run 1 refused {a['refused_gap']:3d}, held-out {a['heldout_last']:.6f}  "
                  f"| run 2 refused {b['refused_gap']:3d}, held-out {b['heldout_last']:.6f}  "
                  f"-> {'IDENTICAL' if same else 'DIFFERENT'}    "
                  f"this seed: {'fires' if a['refused_frac'] >= 0.50 else 'silent'} "
                  f"(refused {a['refused_frac']*100:.0f}% of steps; fire needs >=50% on NOISE, "
                  f"silence needs <=20% on RULE)")
    verdicts_noise = [runs[("noise", s)]["refused_frac"] >= 0.50 for s in (0, 1, 2)]
    verdicts_rule = [runs[("rule", s)]["refused_frac"] <= 0.20 for s in (0, 1, 2)]
    same_verdict = len(set(verdicts_noise)) == 1 and len(set(verdicts_rule)) == 1
    print(f"\n  same seed reproduces the same numbers exactly : {'PASS' if identical else 'FAIL'}")
    print(f"  3 different seeds give the same verdict       : {'PASS' if same_verdict else 'FAIL'}")
    print(f"  per-seed verdicts: NOISE fires {verdicts_noise}, RULE stays silent {verdicts_rule}")
    return {"identical": bool(identical), "same_verdict": bool(same_verdict),
            "noise_verdicts": [bool(v) for v in verdicts_noise],
            "rule_verdicts": [bool(v) for v in verdicts_rule]}


def joint_search():
    """PART 4 - the step size and the gate interact, so calibrate them TOGETHER.

    WHY THIS EXISTS: part 1 was measured with the old step rule (agreement_mode='floor'). When the throttle was
    removed (agreement_mode='one', see lens/scripts/unified_harness_probe.py section 3) part 1 was re-run and
    the winner changed: the same gate that refused 16.3% of the learnable corpus under the throttle refused
    66.0% without it. A gate calibrated against a step size that no longer exists is not calibrated.

    DECISION RULE, FIXED BEFORE RUNNING THIS PART (same binary style as the rest of the file):
      a configuration is ADMISSIBLE iff, for EVERY one of 3 seeds,
          NOISE refused >= 0.50                       (it must fire where memorisation is the only outcome)
          RULE  refused <= 0.20                       (it must not sit on the honest steps)
          RULE held-out with the gate on <= 1.20 x RULE held-out with the gate off   (no damage to learning)
      If several are admissible, the one with the LOWEST RULE refusal wins.
      If NONE is admissible, the result is FAIL and the gate ships switched off, stated plainly - not quietly
      left on with the thresholds moved.
    PREDICTION P16, written before running: with agreement_mode='one' the admissible set is EMPTY (the fast step
    makes the held-out series noisy enough that the same gate refuses 60%+ of honest steps), so the honest
    outcome will be FAIL and the gate off, and the fix is a better detector rather than a retuned threshold.
    """
    print("\n" + "=" * 100)
    print("PART 4 - JOINT SEARCH over the step rule and the gate (decision rule fixed before the run)")
    print("=" * 100)
    print("admissible iff every seed: NOISE >= 0.50, RULE <= 0.20, RULE held-out <= 1.20x gate-off")
    print("prediction P16: with agreement_mode='one' the admissible set is EMPTY -> FAIL -> ship the gate off")
    print()
    grid = list(itertools.product(["one", "floor"], ["ema_band", "divergence"], [0.5, 0.9], [0.0, 0.001, 0.01]))

    # gate-off reference for the RULE corpus, per mode
    rule_ref = {}
    for amode in ("one", "floor"):
        vals = []
        for seed in (0, 1, 2):
            X, y = make_corpus("rule")
            gate = GapGate(variant="off")
            (w, b), log = train_unified(X, y, mode="data", steps=300, lr=0.05, seed=seed, gate=gate,
                                        agreement_mode=amode, verbose=False)
            vals.append(log["heldout_last"])
        rule_ref[amode] = float(np.mean(vals))

    print(f"{'step':<7}{'variant':<12}{'beta':>5}{'tol':>7} | {'NOISE':>16} {'RULE':>16} "
          f"{'RULE vs gate-off':>18} | verdict")
    print("-" * 100)
    admissible = []
    for amode, variant, beta, tol in grid:
        noise, rule = [], []
        for seed in (0, 1, 2):
            noise.append(run_one("noise", variant, beta, tol, seed, agreement_mode=amode))
            rule.append(run_one("rule", variant, beta, tol, seed, agreement_mode=amode))
        nf = np.array([r["refused_frac"] for r in noise])
        rf = np.array([r["refused_frac"] for r in rule])
        rh = float(np.mean([r["heldout_last"] for r in rule]))
        ok = (all(r["refused_frac"] >= 0.50 for r in noise) and all(r["refused_frac"] <= 0.20 for r in rule)
              and rh <= 1.20 * rule_ref[amode])
        if ok:
            admissible.append((rf.mean(), amode, variant, beta, tol))
        print(f"{amode:<7}{variant:<12}{beta:>5}{tol:>7} | {nf.mean():>10.3f}+-{nf.std():.3f} "
              f"{rf.mean():>10.3f}+-{rf.std():.3f} {rh/max(rule_ref[amode],1e-12):>16.3f}x | "
              f"{'ADMISSIBLE' if ok else 'FAIL'}")
    print("-" * 100)
    print(f"gate-off RULE held-out reference: 'one' {rule_ref['one']:.6f}, 'floor' {rule_ref['floor']:.6f}")
    if not admissible:
        print("\nVERDICT: FAIL. No configuration meets the rule fixed above. The gate as parameterised cannot")
        print("tell memorising from learning at any step size tested, so it must NOT ship switched on.")
        best = None
    else:
        admissible.sort()
        best = admissible[0]
        print(f"\nVERDICT: PASS. Best admissible: step '{best[1]}', {best[2]} beta={best[3]} tol={best[4]} "
              f"(RULE refused {best[0]:.3f}), {len(admissible)} admissible configuration(s)")
    print("\nNote on what this does NOT say: it does not say the gate is useless. It says the current DETECTOR")
    print("(a held-out loss EMA compared with a relative band) cannot separate the two corpora at these")
    print("settings. A better detector is the next step, not a moved threshold.")
    return {"rule_ref": rule_ref, "admissible": [[float(a), b, c, float(d), float(e)] for a, b, c, d, e in admissible],
            "best": None if best is None else {"step": best[1], "variant": best[2], "beta": best[3], "tol": best[4]}}


def better_detector():
    """PART 5 - a SECOND detector, judged by the SAME bars (no threshold was moved to make this pass).

    PART 4 found exactly one admissible configuration, and it needed the agreement throttle that section 3 of
    lens/scripts/unified_harness_probe.py had just shown costs the model its ability to generalise
    (0.0% vs 100.0% on unseen pairs). So the old detector ('ema_band': held-out EMA against its own previous
    EMA) is the problem, not the thresholds. This part tests a different detector:

      'best_band': refuse when the smoothed held-out loss rises above the BEST it has ever been.
                   A corpus that can be learned has held-out loss falling, so it sits at its best and the gate
                   stays quiet. A corpus that can only be memorised climbs away from its best immediately.

    SAME BARS AS PART 4 (fixed there, not changed here): admissible iff every seed
        NOISE refused >= 0.50, RULE refused <= 0.20, RULE held-out <= 1.20 x RULE held-out gate-off.

    PREDICTION P17, written before running: 'best_band' with agreement_mode='one' is admissible where
    'ema_band' was not, because the learnable corpus's falling held-out loss keeps it at its own best. I expect
    RULE refusals well under 0.20 with NOISE well over 0.50 at some tolerance, so a configuration passes.
    """
    print("\n" + "=" * 100)
    print("PART 5 - DETECTOR 2 ('best_band': held-out against its own BEST, not its own average)")
    print("=" * 100)
    print("same admission bars as part 4; prediction P17: admissible at agreement_mode='one'")
    print("prediction P17: 'best_band' with the step NOT throttled is admissible, unlike 'ema_band'")
    print()
    grid = list(itertools.product(["one", "floor"], [0.5, 0.9], [0.0, 0.001, 0.01, 0.05]))
    print(f"{'step':<7}{'beta':>5}{'tol':>7} | {'NOISE':>16} {'RULE':>16} {'RULE vs gate-off':>18} | verdict")
    print("-" * 100)
    admissible = []
    rule_ref = {}
    for amode in ("one", "floor"):
        vals = []
        for seed in (0, 1, 2):
            X, y = make_corpus("rule")
            (w, b), log = train_unified(X, y, mode="data", steps=300, lr=0.05, seed=seed,
                                        gate=GapGate(variant="off"), agreement_mode=amode, verbose=False)
            vals.append(log["heldout_last"])
        rule_ref[amode] = float(np.mean(vals))
    for amode, beta, tol in grid:
        noise = [run_one("noise", "best_band", beta, tol, s, agreement_mode=amode) for s in (0, 1, 2)]
        rule = [run_one("rule", "best_band", beta, tol, s, agreement_mode=amode) for s in (0, 1, 2)]
        nf = np.array([r["refused_frac"] for r in noise])
        rf = np.array([r["refused_frac"] for r in rule])
        rh = float(np.mean([r["heldout_last"] for r in rule]))
        ok = (all(r["refused_frac"] >= 0.50 for r in noise) and all(r["refused_frac"] <= 0.20 for r in rule)
              and rh <= 1.20 * rule_ref[amode])
        if ok:
            admissible.append((rf.mean(), amode, beta, tol))
        print(f"{amode:<7}{beta:>5}{tol:>7} | {nf.mean():>10.3f}+-{nf.std():.3f} {rf.mean():>10.3f}+-{rf.std():.3f} "
              f"{rh/max(rule_ref[amode],1e-12):>16.3f}x | {'ADMISSIBLE' if ok else 'FAIL'}")
    print("-" * 100)
    if not admissible:
        print("\nVERDICT: FAIL - detector 2 does not separate the two corpora either. Say so, ship neither on")
        print("default, and leave the memorisation rule as a measured experiment rather than a shipped claim.")
        best = None
    else:
        admissible.sort()
        best = admissible[0]
        print(f"\nVERDICT: PASS. Best admissible: step '{best[1]}', best_band beta={best[2]} tol={best[3]} "
              f"(RULE refused {best[0]:.3f}), {len(admissible)} admissible configuration(s)")
    return {"rule_ref": rule_ref,
            "admissible": [[float(a), b, float(c), float(d)] for a, b, c, d in admissible],
            "best": None if best is None else {"step": best[1], "variant": "best_band", "beta": best[2],
                                               "tol": best[3]}}


def main():
    print("=" * 100)
    print("GAP GATE CALIBRATION - does the memorisation gate fire where it must, and only there?")
    print("=" * 100)
    print("prediction P1: the current 'ema_band' rule refuses < 10% of NOISE steps (needs a 1.5x jump to fire)")
    print("prediction P2: 'divergence' (held-EMA worse AND train-EMA better) fires on NOISE, not on RULE")
    print("verdict rule : PASS iff every seed: NOISE refused >= 0.50 and RULE refused <= 0.20")
    print()

    grid = list(itertools.product(["ema_band", "divergence"], [0.5, 0.9, 0.99], [0.0, 0.001, 0.01, 0.05]))
    seeds = [0, 1, 2]
    rows = []
    t0 = time.time()
    for variant, beta, tol in grid:
        for kind in ("noise", "rule"):
            for seed in seeds:
                rows.append(run_one(kind, variant, beta, tol, seed))
    dt = time.time() - t0

    # ---- aggregate per configuration
    print(f"{'variant':<12}{'beta':>5}{'tol':>8} | {'NOISE refused':>14} {'RULE refused':>13} "
          f"| {'held: noise':>12} {'held: rule':>12} | verdict")
    print("-" * 100)
    results = []
    for variant, beta, tol in grid:
        sel = [r for r in rows if r["variant"] == variant and r["beta"] == beta and r["tol"] == tol]
        noise = [r for r in sel if r["kind"] == "noise"]
        rule = [r for r in sel if r["kind"] == "rule"]
        nf = np.array([r["refused_frac"] for r in noise])
        rf = np.array([r["refused_frac"] for r in rule])
        per_seed_pass = all(
            n["refused_frac"] >= 0.50 and rl["refused_frac"] <= 0.20
            for n, rl in zip(sorted(noise, key=lambda r: r["seed"]), sorted(rule, key=lambda r: r["seed"]))
        )
        verdict = "PASS" if per_seed_pass else "FAIL"
        hn = np.mean([(r["heldout_last"] - r["heldout_first"]) for r in noise])
        hr = np.mean([(r["heldout_last"] - r["heldout_first"]) for r in rule])
        print(f"{variant:<12}{beta:>5}{tol:>8} | {nf.mean():>8.3f}+-{nf.std():.3f} {rf.mean():>8.3f}+-{rf.std():.3f} "
              f"| {hn:>+12.4f} {hr:>+12.4f} | {verdict}")
        results.append({
            "variant": variant, "beta": beta, "tol": tol, "verdict": verdict,
            "noise_refused_mean": float(nf.mean()), "noise_refused_std": float(nf.std()),
            "rule_refused_mean": float(rf.mean()), "rule_refused_std": float(rf.std()),
            "noise_heldout_delta_mean": float(hn), "rule_heldout_delta_mean": float(hr),
        })

    print("-" * 100)
    print(f"{len(rows)} runs of 300 steps in {dt:.1f}s")

    passing = [r for r in results if r["verdict"] == "PASS"]
    band = [r for r in results if r["variant"] == "ema_band" and r["beta"] == 0.9 and r["tol"] == 0.05]
    if band:
        b = band[0]
        print(f"\nTHE SHIPPED RULE (ema_band, beta=0.9, tol=0.05): NOISE refused {b['noise_refused_mean']:.3f}, "
              f"RULE refused {b['rule_refused_mean']:.3f}  -> {b['verdict']}")

    print("\nPASSING CONFIGURATIONS")
    if not passing:
        print("  NONE. The gate as parameterised cannot separate memorisation from learning. FAIL, not 'in spirit'.")
    else:
        passing.sort(key=lambda r: (r["rule_refused_mean"], -r["noise_refused_mean"]))
        for r in passing[:8]:
            print(f"  {r['variant']:<12} beta={r['beta']:<5} tol={r['tol']:<6} "
                  f"noise {r['noise_refused_mean']:.3f}  rule {r['rule_refused_mean']:.3f}")
        best = passing[0]
        print(f"\nBEST: variant={best['variant']} beta={best['beta']} tol={best['tol']}  "
              f"(noise {best['noise_refused_mean']:.3f}, rule {best['rule_refused_mean']:.3f})")

    out = os.path.join(_ROOT, "lens", "runs", "gap_gate_calibration.json")
    part2 = finalist_effect()
    part3 = consistency_check()
    part4 = joint_search()
    part5 = better_detector()
    with open(out, "w") as f:
        json.dump({"prediction": ["P1 current ema_band refuses < 10% on noise",
                                  "P2 divergence fires on noise, not on rule",
                                  "P3 verdict now: the gate as written is decoration",
                                  "P4 on NOISE the calibrated gate ends with a LOWER held-out loss than gate-off",
                                  "P5 on RULE the calibrated gate ends within 20% of gate-off",
                                  "P15 same seed -> identical numbers; different seeds -> the same verdict",
                                  "P16 with agreement_mode='one' the admissible set is EMPTY -> FAIL",
                                  "P17 detector 2 'best_band' is admissible at agreement_mode='one'"],
                   "verdict_rule": "PASS iff every seed: NOISE refused >= 0.50 and RULE refused <= 0.20",
                   "part2_decision_rule": "admissible iff every seed: NOISE>=0.50, RULE<=0.20, RULE held-out <=1.20x gate-off",
                   "part1_grid": {"rows": rows, "results": results},
                   "part2_gate_on_vs_off": part2,
                   "part3_consistency": part3,
                   "part4_joint_search": part4,
                   "part5_better_detector": part5,
                   "shipped_rule": band[0] if band else None,
                   "best": passing[0] if passing else None}, f, indent=1)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
