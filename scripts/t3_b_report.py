"""t3_b_report.py — TARGET B evaluation.

Applies the pre-registered rule from TARGET-B.md:
  * the frozen SELECTION-RULE.md, applied SEPARATELY to each model over its OWN full grid,
    on the 5-seed mean (validation perplexity primary, tie-break accuracy within 1%
    relative, then lower lr)
  * WIN requires readable to beat the normal on BOTH perplexity AND accuracy by more than
    2x pooled std
  * otherwise the verdict is PARITY (within 2 sigma) or BEHIND (worse beyond 2 sigma),
    reported plainly per metric and overall
  * B-3 (audit clauses) is reported separately from the win, because a win without the
    audit clauses is not a Target B win

Nothing here can be tuned by hand: every setting comes from the grid JSONs.
"""
import glob
import json
import math
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")


def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        rows += json.load(open(f))
    return rows


def frozen_select(rows):
    """SELECTION-RULE.md over a model's own grid: group by (lc, lr), mean over seeds."""
    groups = {}
    for r in rows:
        key = (str(r.get("lam_concept", "")), str(r["lr"]))
        groups.setdefault(key, []).append(r)
    cands = []
    for (lc, lr), rs in groups.items():
        n = len(rs)
        cands.append({
            "lc": lc, "lr": lr, "n_seeds": n,
            "ppl": sum(x["val_ppl"] for x in rs) / n,
            "acc": sum(x["val"] for x in rs) / n,
            "params": rs[0]["params"],
            "seeds": sorted(x["seed"] for x in rs),
        })
    # frozen rule: ppl primary, accuracy within 1% RELATIVE as tie-break, then lower lr
    cands.sort(key=lambda c: (round(c["ppl"], 3), -round(c["acc"] / 100.0, 4), float(c["lr"])))
    return cands


read_rows = load("models_b_read_*.json")
norm_rows = load("models_b_norm_*.json")
print("=" * 112)
print("TARGET B — EVALUATION")
print("=" * 112)
print(f"readable runs loaded: {len(read_rows)}   normal runs loaded: {len(norm_rows)}   "
      f"(grid total expected: 60 readable + 20 normal = 80)")

if len(read_rows) < 60 or len(norm_rows) < 20:
    print("\n*** GRID NOT COMPLETE — refusing to score a partial grid. ***")
    print(f"    readable {len(read_rows)}/60, normal {len(norm_rows)}/20")
    sys.exit(2)

rc = frozen_select(read_rows)
nc = frozen_select(norm_rows)

print("\n--- readable grid (5-seed means), sorted by the frozen rule ---")
for c in rc:
    print(f"    lc={c['lc']:<4} lr={c['lr']:<4} seeds={c['n_seeds']}  "
          f"ppl {c['ppl']:8.4f}  acc {c['acc']:6.2f}%  params {c['params']:,}")
print("\n--- normal grid (5-seed means) ---")
for c in nc:
    print(f"    lr={c['lr']:<4} seeds={c['n_seeds']}  ppl {c['ppl']:8.4f}  "
          f"acc {c['acc']:6.2f}%  params {c['params']:,}")

rw, nw = rc[0], nc[0]


def std_of(rows, lc, lr, key):
    v = [x[key] for x in rows
         if str(x.get("lam_concept", "")) == lc and str(x["lr"]) == lr]
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


rp, rps = std_of(read_rows, rw["lc"], rw["lr"], "val_ppl")
ra, ras = std_of(read_rows, rw["lc"], rw["lr"], "val")
np_, nps = std_of(norm_rows, nw["lc"], nw["lr"], "val_ppl")
na, nas = std_of(norm_rows, nw["lc"], nw["lr"], "val")

dppl, dacc = np_ - rp, ra - na                    # positive = readable better
pool_p = math.sqrt(rps ** 2 + nps ** 2)
pool_a = math.sqrt(ras ** 2 + nas ** 2)

print("\n" + "=" * 112)
print("THE SELECTED RUNS (chosen by the frozen rule, not by hand)")
print("=" * 112)
print(f"readable : lc={rw['lc']} lr={rw['lr']}  {rw['n_seeds']} seeds  "
      f"params {rw['params']:,}")
print(f"           acc {ra:.2f} ± {ras:.2f}   ppl {rp:.4f} ± {rps:.4f}")
print(f"normal   : lr={nw['lr']}  {nw['n_seeds']} seeds  params {nw['params']:,}")
print(f"           acc {na:.2f} ± {nas:.2f}   ppl {np_:.4f} ± {nps:.4f}")
print(f"parameter counts: readable {rw['params']:,} vs normal {nw['params']:,} "
      f"(difference {rw['params']-nw['params']:+,})")

print("\n" + "=" * 112)
print("B-1 / B-2 — THE WIN CONDITIONS (> 2x pooled std in readable's favour, BOTH)")
print("=" * 112)


def verdict(delta, pooled, lower_is_better, name):
    bar = 2 * pooled
    if lower_is_better:
        win = delta > bar
        worse = -delta > bar
    else:
        win = delta > bar
        worse = -delta > bar
    v = "WIN" if win else ("BEHIND" if worse else "PARITY")
    print(f"{name}: readable {'better' if delta > 0 else 'worse'} by {abs(delta):.4f} "
          f"| bar = 2 x pooled = {bar:.4f} | VERDICT: {v}")
    return v


v_ppl = verdict(dppl, pool_p, True, "B-1 perplexity")
v_acc = verdict(dacc, pool_a, False, "B-2 accuracy  ")

b3 = None
for f in ("b_audit.json",):
    p = os.path.join(R, f)
    if os.path.exists(p):
        b3 = json.load(open(p))
print("\nB-3 (audit clauses on the winning readable run): "
      + ("see below" if b3 else "NOT YET RUN — requires the audit+probe step"))

print("\n" + "=" * 112)
if v_ppl == "WIN" and v_acc == "WIN" and b3 and b3.get("pass"):
    print("*** TARGET B = MET ***")
elif v_ppl == "WIN" and v_acc == "WIN":
    print("*** readable WINS on both metrics, but B-3 (audit clauses) is not satisfied "
          "-> TARGET B = NOT MET ***")
else:
    print(f"*** TARGET B = NOT MET. Reported plainly: perplexity {v_ppl}, "
          f"accuracy {v_acc}. ***")
    if v_ppl == "PARITY" or v_acc == "PARITY":
        print("    PARITY IS NOT A WIN.")
print("=" * 112)

json.dump({"readable_selected": rw, "normal_selected": nw,
           "readable_grid": rc, "normal_grid": nc,
           "readable": {"acc": ra, "acc_std": ras, "ppl": rp, "ppl_std": rps},
           "normal": {"acc": na, "acc_std": nas, "ppl": np_, "ppl_std": nps},
           "dacc": dacc, "dppl": dppl, "pool_a": pool_a, "pool_p": pool_p,
           "B1_ppl": v_ppl, "B2_acc": v_acc,
           "B_met": bool(v_ppl == "WIN" and v_acc == "WIN" and b3 and b3.get("pass"))},
          open(os.path.join(R, "b_eval.json"), "w"), indent=1)
print("saved runs/t3/b_eval.json")
