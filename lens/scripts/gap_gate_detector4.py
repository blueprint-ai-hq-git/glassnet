"""lens/scripts/gap_gate_detector4.py - detector 4 for the memorisation gate: the LAGGED TREND.

WHY THIS EXISTS
  Detectors 1, 2 and 3 ALL failed their own calibration online (lens/runs/GAP-GATE-CALIBRATION.txt parts 4-5,
  lens/runs/GAP-GATE-DETECTOR3.txt). All three compare a SINGLE STEP's smoothed value against another smoothed
  value, and a single step of a noisy series fires about half the time - including on the learnable corpus once
  it has reached its noise floor. Detector 4 compares the smoothed held-out loss with ITS OWN VALUE 'lag' steps
  ago, so the quantity being judged is a trend over a window instead of one step. At lag=1 it IS detector 1,
  which makes the lag the single variable being tested.

  'trend_band'  refuse when ema_h(now) > ema_h(now - lag) * (1 + tol)
  'gap_trend'   refuse when the gap (train - held) has widened by more than tol * held-out level over the lag

BARS: unchanged from parts 4/5 of lens/scripts/gap_gate_calibration.py (not retuned here):
  every (corpus seed, training seed): NOISE refused >= 0.50, RULE refused <= 0.20,
  and online RULE held-out <= 1.20x the same run with the gate off.

PREDICTIONS: lens/runs/GAP-GATE-DETECTOR4-PREDICTIONS.txt (P35-P37), written before this file was run.
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

from lens.unified import train_unified, GapGate


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the corpora come from the detector-3 script, which takes them from the calibration script itself
_d3 = _load("ggd3", os.path.join("lens", "scripts", "gap_gate_detector3.py"))
make_corpus = _d3.make_corpus
CORPUS_SEEDS, TRAIN_SEEDS, STEPS, LR = _d3.CORPUS_SEEDS, _d3.TRAIN_SEEDS, _d3.STEPS, _d3.LR

OUT = os.path.join(_ROOT, "lens", "runs", "GAP-GATE-DETECTOR4.txt")
OUTJ = os.path.join(_ROOT, "lens", "runs", "gap_gate_detector4.json")
_LINES = []


def say(s=""):
    print(s)
    _LINES.append(s)


def make_grid():
    grid = []
    for variant in ("trend_band", "gap_trend"):
        for lag in (1, 10, 25, 50, 100):
            for tol in (0.0, 0.005, 0.02, 0.05):
                grid.append({"variant": variant, "beta": 0.9, "tol": tol, "ratio": 0.5,
                             "min_rate": 0.002, "lag": lag})
    return grid


def record_traces():
    say("=" * 100)
    say("PART A - RECORD the two losses per step, GATE OFF (same screening data as the detector-3 run)")
    say("=" * 100)
    traces = {}
    for kind in ("noise", "rule"):
        for cseed in CORPUS_SEEDS:
            for tseed in TRAIN_SEEDS:
                X, y = make_corpus(kind, seed=cseed)
                (w, b), log = train_unified(X, y, mode="data", steps=STEPS, lr=LR, seed=tseed,
                                            gate=GapGate(variant="off"), verbose=False, trace=True)
                traces[(kind, cseed, tseed)] = log["trace"]
    say(f"  traces recorded: {len(traces)} runs of {STEPS} steps")
    say("")
    return traces


def replay(traces, row):
    per_seed = {}
    for (kind, cseed, tseed), tr in traces.items():
        gate = GapGate(beta=row["beta"], tol=row["tol"], variant=row["variant"],
                       ratio=row["ratio"], min_rate=row["min_rate"], lag=row["lag"])
        refused = checked = 0
        for rec in tr:
            if rec["by_agreement"]:
                continue
            refuse, ct, ch, _ = gate.evaluate(rec["t"], rec["h"])
            checked += 1
            if refuse:
                refused += 1
            else:
                gate.commit(ct, ch)
        per_seed[(kind, cseed, tseed)] = refused / max(1, checked)
    return per_seed


def run_online(kind, row, tseed, cseed=1234, gate_off=False):
    X, y = make_corpus(kind, seed=cseed)
    if gate_off:
        gate = GapGate(variant="off")
    else:
        gate = GapGate(beta=row["beta"], tol=row["tol"], variant=row["variant"],
                       ratio=row["ratio"], min_rate=row["min_rate"], lag=row["lag"])
    (w, b), log = train_unified(X, y, mode="data", steps=STEPS, lr=LR, seed=tseed, gate=gate, verbose=False)
    checked = max(1, log["refused_gap"] + log["accepted"] + log["refused_agreement"])
    return {"refused_frac": log["refused_gap"] / checked, "refused": log["refused_gap"],
            "accepted": log["accepted"], "heldout_last": log["heldout_last"]}


def offline_screen(traces, grid):
    say("=" * 100)
    say("PART B - OFFLINE PRE-FILTER (a screen only: the detector-3 run measured this replay to be unfaithful,")
    say("         0.185 total deviation against the known online numbers, so PART C takes the verdict)")
    say("=" * 100)
    say(f"{'variant':<12}{'beta':>5}{'lag':>5}{'tol':>7} | {'NOISE refused':>28} {'RULE refused':>28} | screen")
    say("-" * 100)
    rows = []
    for row in grid:
        per = replay(traces, row)
        nf = np.array([v for (k, c, t), v in per.items() if k == "noise"])
        rf = np.array([v for (k, c, t), v in per.items() if k == "rule"])
        ok = bool((nf >= 0.50).all() and (rf <= 0.20).all())
        r = dict(row)
        r.update({"noise_mean": float(nf.mean()), "noise_min": float(nf.min()),
                  "rule_mean": float(rf.mean()), "rule_max": float(rf.max()), "admissible_offline": ok})
        rows.append(r)
        say(f"{row['variant']:<12}{row['beta']:>5}{row['lag']:>5}{row['tol']:>7} | "
            f"{nf.mean():>10.3f}+-{nf.std():.3f}(min {nf.min():.2f}) "
            f"{rf.mean():>10.3f}+-{rf.std():.3f}(max {rf.max():.2f}) | {'admissible' if ok else 'FAIL'}")
    say("-" * 100)
    n_ok = sum(1 for r in rows if r["admissible_offline"])
    say(f"OFFLINE PRE-FILTER: {n_ok} of {len(rows)} configuration(s) look admissible")
    say("")
    return rows


def online_verdict(grid, rows):
    say("=" * 100)
    say("PART C - ONLINE VERDICT: every configuration run for real, gate ON, against the same bars")
    say("=" * 100)
    ref = {}
    for kind in ("noise", "rule"):
        vals = [run_online(kind, None, s, gate_off=True) for s in TRAIN_SEEDS]
        ref[kind] = float(np.mean([v["heldout_last"] for v in vals]))
        say(f"  gate OFF reference  {kind:<6} held-out {ref[kind]:.5f}   (3 training seeds, corpus seed 1234)")
    say("")
    say("bars: NOISE refused >= 0.50 per seed, RULE refused <= 0.20 per seed, RULE held-out <= 1.20x gate-off")
    say("")
    say(f"{'variant':<12}{'beta':>5}{'lag':>5}{'tol':>7} | {'NOISE refused':>22} {'RULE refused':>22} "
        f"| {'RULE vs off':>12} | verdict")
    say("-" * 100)
    off = {(r["variant"], r["beta"], r["tol"], r["lag"]): r for r in rows}
    results = []
    for row in grid:
        per = {kind: [run_online(kind, row, s) for s in TRAIN_SEEDS] for kind in ("noise", "rule")}
        nf = np.array([v["refused_frac"] for v in per["noise"]])
        rf = np.array([v["refused_frac"] for v in per["rule"]])
        nh = float(np.mean([v["heldout_last"] for v in per["noise"]]))
        rh = float(np.mean([v["heldout_last"] for v in per["rule"]]))
        ok = bool((nf >= 0.50).all() and (rf <= 0.20).all() and rh <= 1.20 * ref["rule"])
        res = dict(row)
        res.update({"noise_refused": [float(v) for v in nf], "rule_refused": [float(v) for v in rf],
                    "noise_heldout_mean": nh, "rule_heldout_mean": rh,
                    "rule_vs_gateoff": rh / max(ref["rule"], 1e-12),
                    "noise_vs_gateoff": nh / max(ref["noise"], 1e-12),
                    "offline_noise_mean": off[(row["variant"], row["beta"], row["tol"], row["lag"])]["noise_mean"],
                    "offline_rule_mean": off[(row["variant"], row["beta"], row["tol"], row["lag"])]["rule_mean"],
                    "verdict": "PASS" if ok else "FAIL"})
        results.append(res)
        say(f"{row['variant']:<12}{row['beta']:>5}{row['lag']:>5}{row['tol']:>7} | "
            f"{nf.mean():>8.3f}(min {nf.min():.2f}) {rf.mean():>8.3f}(max {rf.max():.2f}) "
            f"| {res['rule_vs_gateoff']:>11.3f}x | {res['verdict']}")
    say("-" * 100)

    # ---- P36: the lag is the whole difference. lag=1 must reproduce detector 1's known failure.
    say("P36 CHECK - is the lag the whole difference?")
    for r in results:
        if r["variant"] == "trend_band" and r["lag"] == 1 and r["tol"] == 0.0:
            say(f"  trend_band lag=1 tol=0.0 (== detector 1): NOISE {np.mean(r['noise_refused']):.3f}, "
                f"RULE {np.mean(r['rule_refused']):.3f}   |  known detector 1 online: NOISE 0.837, RULE 0.660")
    say("")

    passing = [r for r in results if r["verdict"] == "PASS"]
    if passing:
        passing.sort(key=lambda r: (np.mean(r["rule_refused"]), -np.mean(r["noise_refused"])))
        best = passing[0]
        say(f"ONLINE VERDICT: PASS. {len(passing)} of {len(results)} configuration(s) meet the bars.")
        say(f"  best: {best['variant']} beta={best['beta']} lag={best['lag']} tol={best['tol']}")
        say(f"  NOISE refused {np.mean(best['noise_refused']):.3f} (bar >= 0.50)   "
            f"RULE refused {np.mean(best['rule_refused']):.3f} (bar <= 0.20)   "
            f"RULE held-out {best['rule_vs_gateoff']:.3f}x gate-off (bar <= 1.20x)")
        say(f"  outcome test: NOISE held-out {best['noise_heldout_mean']:.5f} with the gate vs "
            f"{ref['noise']:.5f} without it -> "
            f"{'PASS - refusing prevents the damage' if best['noise_heldout_mean'] < ref['noise'] else 'FAIL - refusing changed nothing'}")
        say(f"  P37: the offline winner survived the online run "
            f"(offline NOISE {best['offline_noise_mean']:.3f} / RULE {best['offline_rule_mean']:.3f} "
            f"vs online {np.mean(best['noise_refused']):.3f} / {np.mean(best['rule_refused']):.3f})")
    else:
        best = None
        say("ONLINE VERDICT: FAIL. No detector-4 configuration meets the bars online. Combined with detectors")
        say("  1-3, the honest conclusion is that a per-step loss gate does not separate these two corpora at")
        say("  this step size, and the memorisation rule stays a measured experiment, not a shipped default.")
    say("")
    return {"gate_off": ref, "results": results, "best": best, "passing": len(passing)}


def consistency(best):
    say("=" * 100)
    say("PART D - IS THE NEW NUMBER CONSISTENT? same seed twice (LEARN-NOT-MEMORISE.md item 1)")
    say("=" * 100)
    if best is None:
        say("  skipped: no configuration met the bars.")
        return {}
    identical = True
    per_seed = {"noise": [], "rule": []}
    for kind in ("noise", "rule"):
        for s in TRAIN_SEEDS:
            a, b = run_online(kind, best, s), run_online(kind, best, s)
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
    say("GAP GATE - DETECTOR 4: the LAGGED TREND (the smoothed loss against its own value 'lag' steps ago)")
    say("=" * 100)
    say("predictions first: lens/runs/GAP-GATE-DETECTOR4-PREDICTIONS.txt (P35-P37), written before this run")
    say("")
    traces = record_traces()
    grid = make_grid()
    rows = offline_screen(traces, grid)
    online = online_verdict(grid, rows)
    cons = consistency(online["best"])
    say(f"total {time.time() - t0:.1f}s")
    with open(OUTJ, "w") as f:
        json.dump({"prediction_file": "lens/runs/GAP-GATE-DETECTOR4-PREDICTIONS.txt",
                   "bars": "NOISE >= 0.50, RULE <= 0.20 per seed; online RULE held-out <= 1.20x gate-off",
                   "corpus_seeds": list(CORPUS_SEEDS), "train_seeds": list(TRAIN_SEEDS),
                   "steps": STEPS, "lr": LR,
                   "part_b_offline": rows, "part_c_online": online, "part_d_consistency": cons}, f, indent=1)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\nwritten: {OUT}")
    print(f"written: {OUTJ}")


if __name__ == "__main__":
    main()
