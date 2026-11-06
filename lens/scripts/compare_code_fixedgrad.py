"""compare_code_fixedgrad.py — the "all levers closed -> capacity-limited" conclusion, re-tested.

Every code arm in the original lab trained under the concept-BCE gradient that was wrong by a
factor of pos_weight (found + fixed 2026-10-10). The conclusion rested on those numbers, so it must
be re-measured. Uses the LAB'S OWN harness (verdict_code_clean.measure: n=150, each checkpoint with
its own vocab, ast.parse + real subprocess, Fisher exact) — one implementation, one construction.

PAIRS (same recipe, only the engine's gradient changed):
   vocab  : covocab32k vs cfix_covocab32k   (baseline code1 vs cfix_code1_6k)
   clean  : cclean32k  vs cfix_cclean32k    (baseline code1 vs cfix_code1_6k)
GATES (as originally pre-registered):
   p2  treatment syntax at n=150 beats the baseline syntax with Fisher exact p < 0.05
   p3  if p2 fails -> data content is not the blocker; code generation is capacity-limited
"""
import math
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import verdict_code_clean as V                     # noqa: E402  (the lab's own harness)


def one_sided_p(a, b, c, d, tail="greater"):
    """Hypergeometric tail: P(X >= a) in the treatment row (a, b) vs baseline (c, d)."""
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    k = a + c
    lo = max(0, k - (c + d))
    hi = min(k, a + b)
    p = 0.0
    for x in range(lo, hi + 1):
        ok = (x >= a) if tail == "greater" else (x <= a)
        if not ok:
            continue
        num = (math.lgamma(a + b + 1) - math.lgamma(x + 1) - math.lgamma(a + b - x + 1)
               + math.lgamma(c + d + 1) - math.lgamma(k - x + 1) - math.lgamma(c + d - (k - x) + 1))
        den = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        p += math.exp(num - den)
    return p


print("=" * 100)
print("CODE CONCLUSION RE-TESTED ON THE CORRECTED GRADIENT (lab's own harness, n=150)")
print("=" * 100)

MODELS = [  # (tag, dtag) — dtag picks the corpus+vocab cache the checkpoint was trained on
    ("code1_ep16",           "v6000_both"),
    ("cfix_code1_6k_ep16",   "v6000_both"),
    ("covocab32k_ep16",      "v32000_code"),
    ("cfix_covocab32k_ep16", "v32000_code"),        # INVALID: retrained on the wrong corpus
    ("cfix_covocab32k_codeonly_ep16", "v32000_code"),  # the corrected vocab arm
    ("cclean32k_ep16",       "v32000_clean"),
    ("cfix_cclean32k_ep16",  "v32000_clean"),
]
CACHE = os.path.join(PROJ, "lens", "runs", "code", "code_fixedgrad_measurements.json")

if os.path.exists(CACHE) and "--remeasure" not in sys.argv:
    res = json.load(open(CACHE))
    print(f"--- loaded cached measurements from {os.path.basename(CACHE)} "
          f"(use --remeasure to redo) ---")
else:
    print("--- measuring 7 checkpoints x ~150 generations (a few minutes) ---")
    res = {}
    for tag, dtag in MODELS:
        m = V.measure(tag, dtag)
        res[tag] = m
        if m:
            print(f"  {tag:<32} V={m['V']:<6} syntax {m['syntax']:>3}/{m['total']}  ran {m['ran']:>3}  "
                  f"unk {m['unk']*100:.2f}%")
        else:
            print(f"  {tag:<32} NOT MEASURED (checkpoint or vocab missing)")
    json.dump(res, open(CACHE, "w"), indent=1)
    print(f"  saved {os.path.basename(CACHE)}")

for tag, dtag in MODELS:            # show what we have either way
    m = res.get(tag)
    if m and (not os.path.exists(CACHE) or "--remeasure" in sys.argv):
        pass
    elif m:
        print(f"  {tag:<32} V={m['V']:<6} syntax {m['syntax']:>3}/{m['total']}  ran {m['ran']:>3}  "
              f"unk {m['unk']*100:.2f}%")
    else:
        print(f"  {tag:<32} not measured")

print("\n" + "-" * 100)

PAIRS = [
    ("VOCAB 6k -> 32k (code-only docs)", "covocab32k_ep16", "cfix_covocab32k_codeonly_ep16",
     "code1_ep16", "cfix_code1_6k_ep16"),
    ("DATA  -> clean corpus (32k)",      "cclean32k_ep16",  "cfix_cclean32k_ep16",
     "code1_ep16", "cfix_code1_6k_ep16"),
]

print("VERDICT — computed, never hardcoded. A lever passes ONLY if the treatment beat its baseline"
      " at p<0.05; if the pass comes from the CONTROL FALLING, that is said out loud, not called a win.")
for name, t_old, t_new, b_old, b_new in PAIRS:
    to, tn, bo, bn = res.get(t_old), res.get(t_new), res.get(b_old), res.get(b_new)
    print(f"\n{name}:")
    if not (to and tn and bo and bn):
        print(f"  INCOMPLETE — measured arms: {[t for t in (t_old, t_new, b_old, b_new) if res.get(t)]}"
              f"  MISSING: {[t for t in (t_old, t_new, b_old, b_new) if not res.get(t)]}")
        continue
    p_old = one_sided_p(to["syntax"], to["total"] - to["syntax"], bo["syntax"], bo["total"] - bo["syntax"])
    p_new = one_sided_p(tn["syntax"], tn["total"] - tn["syntax"], bn["syntax"], bn["total"] - bn["syntax"])
    print(f"  pre-fix : treatment {to['syntax']}/{to['total']}  vs  baseline {bo['syntax']}/{bo['total']}"
          f"   p={p_old:.3f}  -> {'PASS' if p_old < 0.05 else 'FAIL'}")
    print(f"  post-fix: treatment {tn['syntax']}/{tn['total']}  vs  baseline {bn['syntax']}/{bn['total']}"
          f"   p={p_new:.3f}  -> {'PASS' if p_new < 0.05 else 'FAIL'}")
    t_moved = tn["syntax"] - to["syntax"]
    b_moved = bn["syntax"] - bo["syntax"]
    print(f"  mechanism: treatment {t_moved:+d}, baseline {b_moved:+d}")
    if p_new < 0.05 and t_moved <= 0:
        print("  ** PASS DRIVEN BY A FALLING CONTROL — the treatment did not improve. NOT evidence "
              "the lever works. **")
    elif p_new < 0.05:
        print("  treatment improved AND beats baseline — n=150 single seed: replicate before believing.")
    else:
        print("  lever still fails -> it is not the blocker (pre-registered p3 applies).")

print("\nNOTE ON THE VOCAB ARM: tag cfix_covocab32k_ep16 is INVALID and excluded above. That retrain"
      "\nran with CD_VOCAB=32000 but left CODE_DOC at its default 'both', so it trained on a DIFFERENT"
      "\ncorpus (v32000_both) than the original covocab32k (v32000_code) — two things changed at once,"
      "\nwhich the project rule forbids. Its 0/150 is an artifact of the eval feeding it a vocab it"
      "\nnever saw. cfix_covocab32k_codeonly_ep16 is the single-change retrain.")