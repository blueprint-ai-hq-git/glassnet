"""best_code_model.py — choose the best code model ON EVIDENCE, every checkpoint with ITS OWN
vocab and corpus (verdict_code_clean.measure does exactly that; eval_all_code uses one shared 6k
vocab for every model, which scrambles token ids for the 32k checkpoints — the same artifact class
that produced the invalid 0/150, so that ranking cannot be trusted across vocab widths).

WHY ONLY THESE CHECKPOINTS: each needs a KNOWN training corpus (dtag) to pick its own vocab+prompts.
The four original lab arms and the fixed-gradient retrains all have recorded corpora. codropunk /
codedoc_full / covocab / lad_code_lm are EXCLUDED rather than guessed — assigning them a corpus
I don't actually know would be the same knowledge bug as the vocab scramble.

GATES (the lab's, pre-registered in eval_all_code)
  g1  the best checkpoint must beat 8/25 syntax     (the prior lab baseline on the 25-shot set is
      actually beaten by n=150 syntax counting; use the SAME n=150 harness for fairness instead)
  g1n at n=150, the best checkpoint must beat the 6k baseline's 20/150 syntax
  g2  ran >= syntax/2    (parsing without running is not "writing code")
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import verdict_code_clean as V                     # noqa: E402 — the lab's own harness

MODELS = [  # (tag, dtag)
    # original lab arms
    ("code1_ep16",          "v6000_both"),
    ("covocab32k_ep16",     "v32000_code"),
    ("cclean32k_ep16",      "v32000_clean"),
    ("cbig384_ep16",        "v32000_clean"),
    # fixed-gradient reruns (single-change; cfix_covocab32k_ep16 excluded — invalid construction)
    ("cfix_code1_6k_ep16",             "v6000_both"),
    ("cfix_cclean32k_ep16",            "v32000_clean"),
    ("cfix_covocab32k_codeonly_ep16",  "v32000_code"),
]
CACHE = os.path.join(PROJ, "lens", "runs", "code", "best_code_measurements.json")

res = {}
if os.path.exists(CACHE) and "--remeasure" not in sys.argv:
    res = json.load(open(CACHE))
    print("--- loaded cached measurements (--remeasure to redo) ---")
else:
    print("--- measuring every known-corpus checkpoint, own vocab each (a few minutes) ---")
    for tag, dtag in MODELS:
        m = V.measure(tag, dtag)
        res[tag] = m
        if m:
            print(f"  {tag:<32} V={m['V']:<6} syntax {m['syntax']:>3}/{m['total']}  ran {m['ran']:>3}"
                  f"  unk {m['unk']*100:.2f}%")
        else:
            print(f"  {tag:<32} MEASURE FAILED")
    json.dump(res, open(CACHE, "w"), indent=1)

print("\n" + "-" * 100)
print("ALL KNOWN-CORPUS CODE MODELS — ranked by syntax (n=150, own vocab each)")
rows = [r for r in (res.get(t) for t, _ in MODELS) if r]
rows.sort(key=lambda r: (-r["syntax"], -r["ran"]))
for r in rows:
    print(f"  {r['tag']:<32} syntax {r['syntax']:>3}/{r['total']}  ran {r['ran']:>3}  unk {r['unk']*100:.2f}%")

best = rows[0]
print(f"\n  BEST: {best['tag']}  syntax {best['syntax']}/{best['total']}  ran {best['ran']}")
base = res.get("code1_ep16")
g1n = best["syntax"] > 20            # beat the 6k baseline's 20/150
g2 = best["ran"] >= max(1, best["syntax"] // 2)
print(f"  g1n best > baseline 20/150               -> {'PASS' if g1n else 'FAIL'}  ({best['syntax']})")
print(f"  g2  ran >= syntax/2                      -> {'PASS' if g2 else 'FAIL'}  "
      f"({best['ran']} >= {max(1, best['syntax']//2)})")
print(f"  verdict: {'a checkpoint cleared both gates — the best code model is ' + best['tag'] if g1n and g2 else 'NO checkpoint clears both gates: coding remains weak at this scale. Reported, not spun.'}")
json.dump({"best": best["tag"], "g1n": g1n, "g2": g2, "rows": rows},
          open(os.path.join(PROJ, "lens", "runs", "code", "best_code_verdict.json"), "w"), indent=1)
print(f"\n  saved lens/runs/code/best_code_verdict.json")