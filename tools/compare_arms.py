#!/usr/bin/env python3
"""Compare training arms from their own history files, under all three endpoint rules.

WHY THIS EXISTS (2026-11-13): the width claim flipped sign depending on how the endpoint was
picked - best-of-N said +0.87, final epoch said +1.71, last-3-epoch mean said -0.17, on the
SAME two runs. Any claim that quotes only one rule is a claim the author chose the rule for.
So this tool prints all three, always, plus the selection inflation of each arm, and refuses
to print a single headline number.

Usage:
  python3 tools/compare_arms.py lens/runs/chat/d1024_s0_history.jsonl lens/runs/chat/exposure_d512_e9_s0_history.jsonl
  python3 tools/compare_arms.py --key val_asst_acc A.jsonl B.jsonl
"""
import json, sys, statistics as st

def load(path, key="val_asst_acc"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pairs = [(r.get("epoch"), r.get(key)) for r in rows if r.get(key) is not None]
    if not pairs:
        raise SystemExit("%s has no %s column" % (path, key))
    return pairs

def summarise(pairs):
    v = [p[1] * 100 for p in pairs]
    return {"n": len(v), "curve": v, "best": max(v), "best_ep": pairs[v.index(max(v))][0],
            "final": v[-1], "mean": st.mean(v), "last3": st.mean(v[-3:]),
            "inflation": max(v) - st.mean(v)}

def main(argv):
    if "--paired" in argv:
        i = argv.index("--paired")
        rest = argv[i + 1:]
        groups = [[], []]
        cur = 0
        for tok in rest:
            if tok == "--paired":
                cur = 1
                continue
            groups[cur].append(tok)
        return paired_mode(groups)
    key = "val_asst_acc"
    if "--key" in argv:
        i = argv.index("--key"); key = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if len(argv) < 2:
        raise SystemExit(__doc__)
    arms = [(p.split("/")[-1].replace("_history.jsonl", ""), summarise(load(p, key))) for p in argv]
    print("endpoint rules are printed together on purpose: one rule alone lets the author pick the answer\n")
    for name, s in arms:
        print("%s  (%d epochs)" % (name, s["n"]))
        print("   curve: " + " ".join("%d:%.2f" % (e, a) for (e, a) in zip(range(1, s["n"] + 1), s["curve"])))
        print("   best %.2f (ep %d) | final %.2f | mean %.2f | last-3 %.2f | best is %+.2f above its own mean"
              % (s["best"], s["best_ep"], s["final"], s["mean"], s["last3"], s["inflation"]))
    if len(arms) == 2:
        (n1, a), (n2, b) = arms
        print("\n%s MINUS %s under each rule:" % (n1, n2))
        for label, f in (("best-of-N", "best"), ("final epoch", "final"), ("last-3 mean", "last3"), ("all-epoch mean", "mean")):
            d = a[f] - b[f]
            print("   %-15s %+.2f   (%.2f vs %.2f)" % (label, d, a[f], b[f]))
        print("\n   selection inflation: %s %+.2f vs %s %+.2f" % (n1, a["inflation"], n2, b["inflation"]))
        print("   epochs differ (%d vs %d): best-of-N structurally favours the arm with more epochs."
              % (a["n"], b["n"]) if a["n"] != b["n"] else
              "   epochs match (%d): best-of-N is comparable between arms." % a["n"])
    print("\nno single number is printed as the answer - that is the point of this tool")



# --- paired-by-seed mode (added 2026-11-14) -------------------------------------------------------
# The project rule is that seed-matched arms are compared PAIRED ON SEED, never by arm means: the copy
# line got this wrong twice. This mode finds the seed number in each filename, pairs arms by that seed,
# and reports the per-seed difference under every endpoint rule, plus how many seeds agree in sign.
#
#   python3 tools/compare_arms.py --paired A_s0.jsonl A_s1.jsonl --paired B_s0.jsonl B_s1.jsonl
import re as _re

def _seed_of(path):
    m = _re.search(r"_s(\d+)(?:_|\.|$)", path.split("/")[-1])
    return int(m.group(1)) if m else None

def paired_mode(groups):
    """groups: [[armA files], [armB files]] -> paired comparison by seed."""
    a, b = groups[0], groups[1]
    sa = {_seed_of(p): p for p in a if _seed_of(p) is not None}
    sb = {_seed_of(p): p for p in b if _seed_of(p) is not None}
    common = sorted(set(sa) & set(sb))
    if not common:
        raise SystemExit("paired mode needs the same seed number in both arms' filenames")
    rules = ("best", "final", "last3", "mean")
    per_seed = []
    print("paired by seed (%d pair%s):" % (len(common), "" if len(common) == 1 else "s"))
    for s in common:
        A, B = summarise(load(sa[s])), summarise(load(sb[s]))
        row = {r: A[r] - B[r] for r in rules}
        per_seed.append(row)
        print("  seed %d: " % s + "  ".join("%s %+.2f" % (r, row[r]) for r in rules))
    print("\nacross seeds:")
    for r in rules:
        v = [x[r] for x in per_seed]
        if len(v) > 1:
            sd = st.stdev(v)
            se = sd / (len(v) ** 0.5)
            print("  %-8s mean %+.2f  sd %.2f  se %.2f  positive %d/%d%s"
                  % (r, st.mean(v), sd, se, sum(1 for x in v if x > 0), len(v),
                     "   t=%.1f" % (st.mean(v) / se) if se > 0 else ""))
        else:
            print("  %-8s %+.2f  (ONE seed - no spread, no t, quote it as a single run)" % (r, v[0]))
    print("\npaired on seed is the only number quoted for seed-matched arms (project rule).")
    if len(common) < 3:
        print("NOTE: %d pair(s) - with fewer than three seeds no t-statistic is worth reporting." % len(common))


if __name__ == "__main__":
    main(sys.argv[1:])
