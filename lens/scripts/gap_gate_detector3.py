"""lens/scripts/gap_gate_detector3.py - detector 3 for the memorisation gate.

WHY THIS EXISTS
  Detectors 1 ('ema_band') and 2 ('best_band') both failed their own calibration
  (lens/runs/GAP-GATE-CALIBRATION.txt parts 4 and 5): on the corpus that CAN be learned they refused 0.660 and
  0.312 of the steps, against a cost bar of 0.20. Both watch the held-out loss against its own past. On the
  learnable corpus the held-out loss reaches its floor (the y-noise, 0.05^2 = 0.0025) and then stops falling,
  so from then on it is a noisy flat series and "is it worse than its own smoothed self" fires about half the
  time. Convergence and memorisation are indistinguishable to a detector that only watches one of the two
  losses. LEARN-NOT-MEMORISE.md section 2a words the rule as a comparison of the TWO losses.

DETECTOR 3, two variants (both in lens/unified.py GapGate.decide, so the screened thing is the shipped thing):
  'gap_level'  refuse while the training loss sits persistently BELOW the held-out loss by a relative margin.
  'rate_ratio' EMA-smooth both losses, then compare their RELATIVE improvement rates.

METHOD (and why it is honest)
  PART A records, per step, the two losses from runs with the gate OFF (3 corpus seeds x 3 training seeds x 2
  corpora). PART B replays each candidate rule over those recorded series by calling the SHIPPED decision
  function. The replay is first validated against the KNOWN online numbers for the shipped detector, so a
  replay that does not reproduce the known 0.837/0.660 is not trusted. PART C then runs the survivors ONLINE -
  the refusal changes the trajectory, so the offline screen can only be a screen, and the verdict is taken from
  the online run. PART D checks the number is consistent (same seed twice -> identical).

  The admission bars are the ones already fixed in parts 4 and 5 of lens/scripts/gap_gate_calibration.py and
  are NOT retuned here: every (corpus seed, training seed) pair must have
      NOISE refused >= 0.50,  RULE refused <= 0.20,
  and online the RULE final held-out loss with the gate on must be <= 1.20x the same with the gate off.

PREDICTIONS: written before this file was run, in lens/runs/GAP-GATE-DETECTOR3-PREDICTIONS.txt (P31-P34).
"""
import importlib.util
import itertools
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.unified import train_unified, GapGate  # the SHIPPED code

# the corpora are taken from the calibration script itself, so the two tests cannot drift apart
_spec = importlib.util.spec_from_file_location(
    "ggc", os.path.join(_ROOT, "lens", "scripts", "gap_gate_calibration.py"))
_ggc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ggc)
make_corpus = _ggc.make_corpus

CORPUS_SEEDS = (1234, 7, 99)
TRAIN_SEEDS = (0, 1, 2)
STEPS = 300
LR = 0.05
OUT = os.path.join(_ROOT, "lens", "runs", "GAP-GATE-DETECTOR3.txt")
OUTJ = os.path.join(_ROOT, "lens", "runs", "gap_gate_detector3.json")

_LINES = []


def say(s=""):
    print(s)
    _LINES.append(s)


def record_traces():
    """PART A - record the two losses every step with the gate OFF. This is the screening data."""
    say("=" * 100)
    say("PART A - RECORD the two losses per step, GATE OFF (the screening data; nothing is judged here)")
    say("=" * 100)
    traces = {}
    summary = {}
    for kind in ("noise", "rule"):
        ratios, t_last, h_last = [], [], []
        for cseed in CORPUS_SEEDS:
            for tseed in TRAIN_SEEDS:
                X, y = make_corpus(kind, seed=cseed)
                (w, b), log = train_unified(X, y, mode="data", steps=STEPS, lr=LR, seed=tseed,
                                            gate=GapGate(variant="off"), verbose=False, trace=True)
                traces[(kind, cseed, tseed)] = log["trace"]
                t_last.append(log["train_last"])
                h_last.append(log["heldout_last"])
        summary[kind] = {"train_last_mean": float(np.mean(t_last)), "heldout_last_mean": float(np.mean(h_last)),
                         "ratio_mean": float(np.mean([t / h for t, h in zip(t_last, h_last)]))}
    say(f"  traces recorded: {len(traces)} runs of {STEPS} steps "
        f"({len(CORPUS_SEEDS)} corpus seeds x {len(TRAIN_SEEDS)} training seeds x 2 corpora)")
    say("")
    say("  P31 CHECK - the two-loss LEVEL at the end of the run, gate off (mean over the 9 runs per corpus):")
    say(f"    NOISE  train {summary['noise']['train_last_mean']:.5f}   held-out "
        f"{summary['noise']['heldout_last_mean']:.5f}   ratio train/held {summary['noise']['ratio_mean']:.3f}")
    say(f"    RULE   train {summary['rule']['train_last_mean']:.5f}   held-out "
        f"{summary['rule']['heldout_last_mean']:.5f}   ratio train/held {summary['rule']['ratio_mean']:.3f}")
    say("    P31 said NOISE ratio < 0.2 and RULE ratio > 0.4. Read the two numbers above against that.")
    say("")
    return traces, summary


def replay(traces, row):
    """PART B helper - replay one rule over the recorded series through the SHIPPED decision function.
    Only accepted steps move the gate's state, exactly as in the online loop."""
    per_seed = {}
    for (kind, cseed, tseed), tr in traces.items():
        gate = GapGate(beta=row["beta"], tol=row["tol"], variant=row["variant"],
                       ratio=row["ratio"], min_rate=row["min_rate"])
        refused = checked = 0
        for rec in tr:
            if rec["by_agreement"]:
                continue                      # the gate never saw these steps online either
            refuse, ct, ch, _ = gate.evaluate(rec["t"], rec["h"])
            checked += 1
            if refuse:
                refused += 1
            else:
                gate.commit(ct, ch)
        per_seed[(kind, cseed, tseed)] = refused / max(1, checked)
    return per_seed


def make_grid():
    """Every candidate rule, fully specified (no None anywhere): gap_level uses ratio/min_rate defaults,
    rate_ratio uses tol=0.0. The grid is declared here, before any number is printed."""
    grid = []
    for beta, tol in itertools.product((0.5, 0.9), (0.1, 0.2, 0.4, 0.6)):
        grid.append({"variant": "gap_level", "beta": beta, "tol": tol, "ratio": 0.5, "min_rate": 0.002})
    for beta, ratio, min_rate in itertools.product((0.5, 0.9), (0.25, 0.5, 0.75), (0.002, 0.01)):
        grid.append({"variant": "rate_ratio", "beta": beta, "tol": 0.0, "ratio": ratio, "min_rate": min_rate})
    return grid


def offline_screen(traces, grid):
    """PART B - every candidate rule, judged by the bars that were fixed in parts 4/5 (not moved here)."""
    say("=" * 100)
    say("PART B - OFFLINE SCREEN over the recorded series (a SCREEN only: PART C takes the verdict)")
    say("=" * 100)
    say("bars (fixed in parts 4/5 of gap_gate_calibration.py, unchanged here):")
    say("  every (corpus seed, training seed): NOISE refused >= 0.50 and RULE refused <= 0.20")
    say("")

    # ---- REPLAY VALIDATION: the replay must reproduce the KNOWN online numbers of the shipped detector.
    # Known, measured online in lens/runs/GAP-GATE-CALIBRATION.txt part 1: ema_band beta=0.5 tol=0.0 gives
    # NOISE 0.837+-0.009 and RULE 0.660+-0.024 (corpus seed 1234, 3 training seeds). If the replay does not
    # land near those, the replay is wrong and nothing below it can be believed.
    val = replay(traces, {"variant": "ema_band", "beta": 0.5, "tol": 0.0, "ratio": 0.5, "min_rate": 0.002})
    nv = [v for (k, c, t), v in val.items() if k == "noise" and c == 1234]
    rv = [v for (k, c, t), v in val.items() if k == "rule" and c == 1234]
    say("REPLAY VALIDATION (the shipped detector, beta=0.5 tol=0.0, corpus seed 1234)")
    say(f"  replay says  NOISE {np.mean(nv):.3f}  RULE {np.mean(rv):.3f}")
    say(f"  known online NOISE 0.837  RULE 0.660   (lens/runs/GAP-GATE-CALIBRATION.txt part 1)")
    dev = abs(np.mean(nv) - 0.837) + abs(np.mean(rv) - 0.660)
    say(f"  total deviation {dev:.3f}  -> {'REPLAY TRUSTWORTHY (within 0.10)' if dev <= 0.10 else 'REPLAY NOT TRUSTWORTHY'}")
    say("  This is the P34 caveat measured, not argued: a rule changes the trajectory it is judged on, so the")
    say("  screen below is a pre-filter and the verdict is taken from the ONLINE run in PART C.")
    say("")

    say(f"{'variant':<12}{'beta':>5}{'tol':>6}{'ratio':>7}{'min_rate':>9} | "
        f"{'NOISE refused':>28} {'RULE refused':>28} | screen")
    say("-" * 100)
    rows = []
    admissible = []
    for row in grid:
        per = replay(traces, row)
        nf = np.array([v for (k, c, t), v in per.items() if k == "noise"])
        rf = np.array([v for (k, c, t), v in per.items() if k == "rule"])
        ok = bool((nf >= 0.50).all() and (rf <= 0.20).all())
        r = dict(row)
        r.update({"noise_mean": float(nf.mean()), "noise_min": float(nf.min()),
                  "rule_mean": float(rf.mean()), "rule_max": float(rf.max()),
                  "admissible_offline": ok})
        rows.append(r)
        if ok:
            admissible.append(r)
        say(f"{row['variant']:<12}{row['beta']:>5}{row['tol']:>6}{row['ratio']:>7}{row['min_rate']:>9} | "
            f"{nf.mean():>10.3f}+-{nf.std():.3f}(min {nf.min():.2f}) "
            f"{rf.mean():>10.3f}+-{rf.std():.3f}(max {rf.max():.2f}) | {'admissible' if ok else 'FAIL'}")
    say("-" * 100)
    admissible.sort(key=lambda r: r["rule_mean"])
    if admissible:
        a = admissible[0]
        say(f"OFFLINE PRE-FILTER: {len(admissible)} configuration(s) look admissible; lowest RULE refusal "
            f"{a['variant']} beta={a['beta']} tol={a['tol']} ratio={a['ratio']} min_rate={a['min_rate']} "
            f"(RULE {a['rule_mean']:.3f}, NOISE {a['noise_mean']:.3f})")
    else:
        say("OFFLINE PRE-FILTER: none look admissible. PART C still runs every configuration online, because")
        say("the screen itself was just measured as unfaithful (above).")
    say("")
    return rows, admissible


def run_online(kind, row, tseed, cseed=1234, gate_off=False):
    X, y = make_corpus(kind, seed=cseed)
    if gate_off:
        gate = GapGate(variant="off")
    else:
        gate = GapGate(beta=row["beta"], tol=row["tol"], variant=row["variant"],
                       ratio=row["ratio"], min_rate=row["min_rate"])
    (w, b), log = train_unified(X, y, mode="data", steps=STEPS, lr=LR, seed=tseed, gate=gate, verbose=False)
    checked = max(1, log["refused_gap"] + log["accepted"] + log["refused_agreement"])
    return {"refused_frac": log["refused_gap"] / checked, "refused": log["refused_gap"],
            "accepted": log["accepted"], "heldout_last": log["heldout_last"], "train_last": log["train_last"]}


def online_verdict(grid, rows, admissible):
    """PART C - run EVERY configuration ONLINE. This is the verdict: refusing changes the trajectory the rule
    is then judged on, so the offline screen cannot decide it (prediction P34, measured in PART B)."""
    say("=" * 100)
    say("PART C - ONLINE VERDICT: every configuration run for real, gate ON, against the same bars")
    say("=" * 100)
    ref = {}
    for kind in ("noise", "rule"):
        vals = [run_online(kind, None, s, gate_off=True) for s in TRAIN_SEEDS]
        ref[kind] = {"heldout_mean": float(np.mean([v["heldout_last"] for v in vals])),
                     "heldout_all": [v["heldout_last"] for v in vals]}
        say(f"  gate OFF reference  {kind:<6} held-out {ref[kind]['heldout_mean']:.5f}   "
            f"(3 training seeds, corpus seed 1234)")
    say("")
    say("bars: NOISE refused >= 0.50 per seed, RULE refused <= 0.20 per seed, "
        "RULE held-out <= 1.20x gate-off")
    say("")
    say(f"{'variant':<12}{'beta':>5}{'tol':>6}{'ratio':>7}{'min_rate':>9} | "
        f"{'NOISE refused':>22} {'RULE refused':>22} | {'RULE vs off':>12} | verdict")
    say("-" * 100)
    results = []
    off_by_key = {(r["variant"], r["beta"], r["tol"], r["ratio"], r["min_rate"]): r for r in rows}
    for row in grid:
        per = {}
        for kind in ("noise", "rule"):
            per[kind] = [run_online(kind, row, s) for s in TRAIN_SEEDS]
        nf = np.array([v["refused_frac"] for v in per["noise"]])
        rf = np.array([v["refused_frac"] for v in per["rule"]])
        rh = float(np.mean([v["heldout_last"] for v in per["rule"]]))
        nh = float(np.mean([v["heldout_last"] for v in per["noise"]]))
        ok = bool((nf >= 0.50).all() and (rf <= 0.20).all() and rh <= 1.20 * ref["rule"]["heldout_mean"])
        key = (row["variant"], row["beta"], row["tol"], row["ratio"], row["min_rate"])
        res = dict(row)
        res.update({"noise_refused": [float(v) for v in nf], "rule_refused": [float(v) for v in rf],
                    "noise_heldout_mean": nh, "rule_heldout_mean": rh,
                    "rule_vs_gateoff": rh / max(ref["rule"]["heldout_mean"], 1e-12),
                    "noise_vs_gateoff": nh / max(ref["noise"]["heldout_mean"], 1e-12),
                    "offline_noise_mean": off_by_key[key]["noise_mean"],
                    "offline_rule_mean": off_by_key[key]["rule_mean"],
                    "verdict": "PASS" if ok else "FAIL"})
        results.append(res)
        say(f"{row['variant']:<12}{row['beta']:>5}{row['tol']:>6}{row['ratio']:>7}{row['min_rate']:>9} | "
            f"{nf.mean():>8.3f}(min {nf.min():.2f}) {rf.mean():>8.3f}(max {rf.max():.2f}) "
            f"| {res['rule_vs_gateoff']:>11.3f}x | {res['verdict']}")
    say("-" * 100)
    passing = [r for r in results if r["verdict"] == "PASS"]
    if passing:
        passing.sort(key=lambda r: np.mean(r["rule_refused"]))
        best = passing[0]
        say(f"ONLINE VERDICT: PASS. {len(passing)} of {len(results)} configuration(s) meet the bars.")
        say(f"  best: {best['variant']} beta={best['beta']} tol={best['tol']} ratio={best['ratio']} "
            f"min_rate={best['min_rate']}")
        say(f"  NOISE refused {np.mean(best['noise_refused']):.3f} (bar >= 0.50)   "
            f"RULE refused {np.mean(best['rule_refused']):.3f} (bar <= 0.20)   "
            f"RULE held-out {best['rule_vs_gateoff']:.3f}x gate-off (bar <= 1.20x)")
        say(f"  outcome test: NOISE held-out {best['noise_heldout_mean']:.5f} with the gate vs "
            f"{ref['noise']['heldout_mean']:.5f} without it -> "
            f"{'PASS - refusing prevents the damage' if best['noise_heldout_mean'] < ref['noise']['heldout_mean'] else 'FAIL - refusing changed nothing'}")
        say(f"  P34 MEASURED - offline screen vs online run for the winner: NOISE "
            f"{best['offline_noise_mean']:.3f} offline vs {np.mean(best['noise_refused']):.3f} online; "
            f"RULE {best['offline_rule_mean']:.3f} offline vs {np.mean(best['rule_refused']):.3f} online")
    else:
        best = None
        say("ONLINE VERDICT: FAIL. No detector-3 configuration meets the bars online. Stated plainly: the")
        say("  memorisation gate is still not calibrated on these corpora, and it stays a measured experiment.")
    say("")
    return {"gate_off": ref, "results": results, "best": best, "passing": len(passing)}


def consistency(best):
    """PART D - LEARN-NOT-MEMORISE.md item 1 on the NEW rule: same seed twice -> identical numbers."""
    say("=" * 100)
    say("PART D - IS THE NEW NUMBER CONSISTENT? same seed twice, then 3 seeds' verdicts")
    say("=" * 100)
    if best is None:
        say("  skipped: no configuration met the bars, so there is no new rule to check.")
        return {}
    identical = True
    per_seed = {"noise": [], "rule": []}
    for kind in ("noise", "rule"):
        for s in TRAIN_SEEDS:
            a = run_online(kind, best, s)
            b = run_online(kind, best, s)
            same = (a["refused"] == b["refused"] and a["accepted"] == b["accepted"]
                    and a["heldout_last"] == b["heldout_last"])
            identical = identical and same
            per_seed[kind].append(a["refused_frac"])
            say(f"  {kind:<6} seed {s}: refused {a['refused']:3d} / held-out {a['heldout_last']:.6f}  "
                f"| refused {b['refused']:3d} / held-out {b['heldout_last']:.6f}  "
                f"-> {'IDENTICAL' if same else 'DIFFERENT'}")
    vn = [f >= 0.50 for f in per_seed["noise"]]
    vr = [f <= 0.20 for f in per_seed["rule"]]
    say(f"\n  same seed reproduces the same numbers exactly : {'PASS' if identical else 'FAIL'}")
    say(f"  per-seed verdicts: NOISE fires {vn}, RULE stays silent {vr}")
    return {"identical": bool(identical), "noise_verdicts": vn, "rule_verdicts": vr}


def main():
    t0 = time.time()
    say("=" * 100)
    say("GAP GATE - DETECTOR 3 ('gap_level' and 'rate_ratio': the rule compares the TWO losses)")
    say("=" * 100)
    say("predictions first: lens/runs/GAP-GATE-DETECTOR3-PREDICTIONS.txt (P31-P34), written before this run")
    say("")
    traces, summary = record_traces()
    grid = make_grid()
    rows, admissible = offline_screen(traces, grid)
    online = online_verdict(grid, rows, admissible)
    cons = consistency(online["best"])
    say(f"total {time.time() - t0:.1f}s")

    with open(OUTJ, "w") as f:
        json.dump({"prediction_file": "lens/runs/GAP-GATE-DETECTOR3-PREDICTIONS.txt",
                   "bars": "every (corpus seed, training seed): NOISE >= 0.50, RULE <= 0.20; "
                           "online RULE held-out <= 1.20x gate-off",
                   "corpus_seeds": list(CORPUS_SEEDS), "train_seeds": list(TRAIN_SEEDS),
                   "steps": STEPS, "lr": LR,
                   "part_a_signal_summary": summary,
                   "part_b_offline_screen": rows,
                   "part_c_online": online,
                   "part_d_consistency": cons}, f, indent=1)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\nwritten: {OUT}")
    print(f"written: {OUTJ}")


if __name__ == "__main__":
    main()
