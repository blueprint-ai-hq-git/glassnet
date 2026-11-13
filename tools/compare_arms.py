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

if __name__ == "__main__":
    main(sys.argv[1:])
