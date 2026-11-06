"""loop_step.py — run ONE loop iteration (one change, single variable).

Sequence (mission-compliant):
  1. write hypothesis + success test to the iteration folder (BEFORE run)
  2. load the fixed split (test quarantined) + the named concept set cache
  3. readable model, 3 seeds, auto-pilot, evaluated on VALIDATION only
  4. same-size normal net, 3 seeds (both param counts reported)
  5. verdict vs CURRENT-BEST: keep only if the win beats seed noise AND the
     new concept set passes the readability audit (--audit)
  6. append LOG.md, save all artifacts in runs/loop/iterNNN/
  7. --milestone: touch the quarantined TEST once (logged), check Rung-1 gate

Exit codes: 0 ok · 42 RUNGS-GATE-PASSED (write STOPPED.md and stop)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.loop_lib import (HERE, RUNS, load_split, load_concept_set,  # noqa: E402
                               readable_benchmark, normal_benchmark,
                               readable_params, normal_params, match_hidden,
                               promote_check, load_current_best,
                               save_current_best, log_test_use, log_line)
from glassnet.eval import accuracy as acc_fn  # noqa: E402

LOG_PATH = os.path.join(HERE, "LOG.md")


def append_log(row):
    with open(LOG_PATH, "a") as f:
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--names", required=True)          # comma-separated
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--success", required=True)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--concept-hidden", type=int, default=0)
    ap.add_argument("--rule-linear", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--task", default="fashion-mnist",
                    choices=["fashion-mnist", "pathmnist", "banking77"])
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--input-dim", type=int, default=784)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--milestone", default="")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",")]
    iterdir = os.path.join(RUNS, f"iter{args.iter:03d}")
    os.makedirs(iterdir, exist_ok=True)

    # ---- 1. hypothesis + success test, written BEFORE any training --------
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    hyp = {"iteration": args.iter, "written_before_run": when,
           "hypothesis": args.hyp, "success_test": args.success,
           "names": names, "K": len(names), "R": args.R}
    with open(os.path.join(iterdir, "hypothesis.json"), "w") as f:
        json.dump(hyp, f, indent=1)
    with open(os.path.join(iterdir, "hypothesis.md"), "w") as f:
        f.write(f"# iteration {args.iter}\n\n**hypothesis (written "
                f"{when}, BEFORE the run):**\n{args.hyp}\n\n"
                f"**success test:**\n{args.success}\n\n"
                f"concept set ({len(names)}): {', '.join(names)}\n"
                f"rules R = {args.R}\n")
    log_line(f"[iter {args.iter}] hypothesis written -> "
             f"runs/loop/iter{args.iter:03d}/hypothesis.md")

    # ---- 2. data -----------------------------------------------------------
    Xtr, ytr, Xval, yval, Xte, yte = load_split(args.task)
    Ctr, Cval, Cte, _, _ = load_concept_set(Xtr, ytr, Xval, yval, Xte, yte,
                                            names, task=args.task)

    # ---- 3. readable, 3 seeds, validation only -----------------------------
    t0 = time.time()
    rvals, rinfos, rnets = readable_benchmark(Xtr, ytr, Ctr, Xval, yval,
                                              Cval, names, R=args.R,
                                              concept_hidden=args.concept_hidden,
                                              rule_linear=args.rule_linear,
                                              alpha=args.alpha,
                                              no_rules=args.no_rules,
                                              classes=args.classes,
                                              d_in=args.input_dim,
                                              epochs=args.epochs)
    for s in rinfos:
        rnets[s].save(os.path.join(iterdir, f"final_seed{s}.npz"))
        json.dump(rinfos[s], open(os.path.join(iterdir, f"info_seed{s}.json"),
                                  "w"), indent=1)
    rm, rs = float(np.mean(list(rvals.values()))), float(np.std(list(rvals.values())))

    # ---- 4. same-size normal net, 3 seeds ----------------------------------
    r_params = readable_params(len(names), args.R, args.concept_hidden,
                               args.no_rules, classes=args.classes,
                               d_in=args.input_dim)
    hidden = match_hidden(r_params, classes=args.classes, d_in=args.input_dim)
    nvals, nnets = normal_benchmark(Xtr, ytr, Xval, yval, hidden,
                                    classes=args.classes, d_in=args.input_dim,
                                    epochs=args.epochs)
    for s in nnets:
        nnets[s].save(os.path.join(iterdir, f"normal_seed{s}.npz"))
    nm, ns = float(np.mean(list(nvals.values()))), float(np.std(list(nvals.values())))
    n_params = normal_params(hidden, classes=args.classes, d_in=args.input_dim)

    # ---- 5. verdict vs CURRENT-BEST (task-scoped) --------------------------
    cb = load_current_best(args.task)
    if cb is None:
        verdict, delta, noise = "BASELINE", float("nan"), float("nan")
    else:
        verdict, rm2, rs2, delta, noise = promote_check(rvals, cb["rval_vals"])
    audit_pass = None
    if args.audit:
        from glassnet.loop_audit import audit_table, train_lie_net
        model0 = rnets[0]
        if len(model0.concept_names) != len(names):
            model0.concept_names = list(names)
        rows, matrix = audit_table(model0, Xval, Cval, names)
        prev = (load_current_best(args.task) or {}).get("names") or []
        new_names = [n for n in names if n not in prev]
        lie, lie_k, steal = train_lie_net(Xtr, ytr, Ctr, Xval, yval, Cval,
                                          names, classes=args.classes)
        lrows, lmatrix = audit_table(lie, Xval, Cval, names)
        honest_own = rows[lie_k]["fidelity"]
        lie_own = lrows[lie_k]["fidelity"]
        lie_track = float(lmatrix[lie_k, steal]) if steal >= 0 else float("nan")
        caught = (lie_own < 0.55) or (lie_own < honest_own - 0.15)
        broken = [(r["name"], r["fidelity"]) for r in rows
                  if not (r["fidelity"] >= 0.55)]
        new_broken = [n for n in new_names
                      if not (rows[names.index(n)]["fidelity"] >= 0.55)]
        audit_pass = caught and (not new_broken)
        lie_src = ("shuffled own targets" if steal < 0
               else f"'{names[steal]}' targets")
        lie_eff = ("separated from its name by construction" if steal < 0
                   else f"secretly tracks {names[steal]} at {lie_track:.3f}")
        verdict_lie = "CAUGHT ✔" if caught else "MISSED ✘"
        L = ["# LOOP AUDIT", "",
             f"set ({len(names)}): {', '.join(names)}", "",
             f"new concepts vs previous set: {new_names if new_names else 'none'}", "",
             "| neuron | fidelity | pos | leans toward | flags |",
             "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['name']} | {r['fidelity']:.3f} | "
                     f"pos {r.get('pos_frac', float('nan')):.2f} | "
                     f"{r['leans_to']} ({r['leans_auroc']:.2f}) | {r['flags']} |")
        L += ["",
              f"broken (fidelity<0.55): {broken if broken else 'none'}",
              f"broken among NEW concepts: {new_broken if new_broken else 'none'}",
              "",
              f"LIE CONTROL: '{names[lie_k]}' secretly trained on "
              f"{lie_src}: "
              f"honest-name fidelity {honest_own:.3f} -> "
              f"{lie_own:.3f} (lost {honest_own-lie_own:.3f}), "
              f"{lie_eff}; {verdict_lie}",
              "",
              f"AUDIT: {'PASS ✔' if audit_pass else 'FAIL ✘'} "
              f"(lie {'caught' if caught else 'missed'}"
              f"{', new-concept breakage: ' + str(new_broken) if new_broken else ''})",
              ""]
        with open(os.path.join(iterdir, "audit.md"), "w") as f:
            f.write("\n".join(L) + "\n")
        print("\n".join(L))
        if cb is not None and verdict == "KEEP" and not audit_pass:
            log_line(f"[iter {args.iter}] READABILITY GATE FAILED -> forced "
                     f"REVERT (new concepts not trustworthy)")
            verdict = "REVERT"

    # ---- 6. promote / record -----------------------------------------------
    entry = {"iteration": args.iter, "task": args.task, "names": names,
             "K": len(names), "classes": args.classes, "input_dim": args.input_dim,
             "R": args.R, "epochs": args.epochs, "rval_vals": {str(s): v for s, v in rvals.items()},
             "rval_mean": rm, "rval_std": rs,
             "nval_vals": {str(s): v for s, v in nvals.items()},
             "nval_mean": nm, "nval_std": ns,
             "readable_params": r_params, "normal_params": n_params,
             "normal_hidden": hidden, "audit_pass": audit_pass,
             "verdict": verdict, "delta_vs_best": delta, "noise": noise,
             "time": time.time() - t0}
    json.dump(entry, open(os.path.join(iterdir, "results.json"), "w"), indent=1)
    if verdict == "KEEP" or cb is None:
        save_current_best(entry)
        log_line(f"[iter {args.iter}] PROMOTED to CURRENT-BEST "
                 f"(readable val {rm*100:.2f}% ± {rs*100:.2f}%)")

    row = (f"| {args.iter:03d} | {args.hyp[:60].replace('|', '/')} | "
           f"{rm*100:.2f}±{rs*100:.2f} | {nm*100:.2f}±{ns*100:.2f} | "
           f"{r_params} | {n_params} | "
           f"{(delta if not np.isnan(delta) else 0)*100:+.2f} pts "
           f"(noise {noise*100:.2f}) | {verdict}"
           f"{' AUDIT-PASS' if audit_pass else (' AUDIT-FAIL' if audit_pass is False else '')} |")
    append_log(row)
    log_line(f"[iter {args.iter}] verdict: {verdict} | readable val "
             f"{rm*100:.2f}±{rs*100:.2f} | normal val {nm*100:.2f}±{ns*100:.2f} "
             f"| params {r_params} vs {n_params}")

    # ---- 7. milestone: touch quarantined test once (logged) ----------------
    if args.milestone:
        best = load_current_best()
        # evaluate promoted readable (rebuild from best snapshot seeds) and
        # the same-size normal net on TEST once
        from glassnet.phase2_model import GlassNetPhase2
        test_r = []
        for s in (0, 1, 2):
            m = GlassNetPhase2.load(os.path.join(iterdir, f"final_seed{s}.npz"))
            test_r.append(acc_fn(m, Xte, yte))
        test_n = [acc_fn(nnets[s], Xte, yte) for s in (0, 1, 2)]
        trm, trs = float(np.mean(test_r)), float(np.std(test_r))
        tnm, tns = float(np.mean(test_n)), float(np.std(test_n))
        numbers = (f"{args.milestone}: readable test {trm*100:.2f}% ± "
                   f"{trs*100:.2f}% vs same-size normal {tnm*100:.2f}% ± "
                   f"{tns*100:.2f}% (params {r_params} vs {n_params})")
        log_test_use(time.strftime("%Y-%m-%d %H:%M"), numbers)
        json.dump({"milestone": args.milestone, "readable_test": test_r,
                   "normal_test": test_n},
                  open(os.path.join(iterdir, "milestone.json"), "w"), indent=1)
        gate_pass = (abs(trm - tnm) <= 2.0) and (trm >= tnm)
        if gate_pass:
            with open(os.path.join(HERE, "STOPPED.md"), "w") as f:
                f.write(f"# STOPPED — Rung-1 gate passed\n\n{time.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"{numbers}\n\nRung 2 (CIFAR-10 etc.) NOT started — stop "
                        f"condition per mission: a rung's gate passes; wait for "
                        f"Yahya before the next rung.\n")
            log_line(f"[iter {args.iter}] *** RUNG-1 GATE PASSED — stopping ***")
            return 42
        log_line(f"[iter {args.iter}] milestone checked: gate NOT passed "
                 f"(Δ {abs(trm-tnm)*100:.2f} pts, readable "
                 f"{'behind' if trm < tnm else 'ahead'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())