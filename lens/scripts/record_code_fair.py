"""record_code_fair.py — the fair comparison, and the correction to my own reading.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "code")


def stat(name):
    f = os.path.join(P, f"{name}_log.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    h = d["history"][-1]
    return {"ppl": h["val_ppl"], "acc": h["val_acc"] * 100, "n": len(d["history"]),
            "params": d.get("params", 0),
            "concepts": (d.get("concepts", {}).get("n_kept"), d.get("concepts", {}).get("n_defined"))}


rx = json.load(open(os.path.join(P, "code1_reexec.json")))["stats"]
c1, cd, cf = stat("code1"), stat("codedoc"), stat("codedoc_full")

LOG = f"""
---

## 2026-10-08 — CODE: THE UNFAIR COMPARISON CORRECTED, AND THE CORPUS CHANGE CONFIRMED

### WHAT I GOT WRONG

First code-only run reported **2/25 syntax against 4/25**, which reads as the change being harmful.
It was not a comparison:
```
  codedoc  code-only   4 epochs   120,000 windows   ppl 91.06  acc 28.97%
  code1    both       16 epochs 1,200,000 windows   ppl 44.26  acc 36.44%
```
**40x less training.** The pair compares training BUDGET, not the corpus change. Same failure as the
big-ladder bug (two runs of different sizes presented as a comparison), and this time both runs were
mine. My pre-registered prediction -- *"syntax improves from 4/25 but stays LOW"* -- was left
**untested** rather than declared wrong, because the run meant to test it was under-budget.

### THE FAIR RUN

Identical to code1 in every setting -- d=192, window=16, epochs=16, batch=256, lr=0.4, lr-decay=0.5,
lam-concept=0.3, max-windows=1,200,000, device=gpu, seed=0. **Only `CODE_DOC` differs.**

```
  run              corpus   epochs     windows      ppl     acc     syntax
  code1            both         16   1,200,000    44.26   36.44%     4/25
  codedoc (UNFAIR) code          4     120,000    91.06   28.97%     2/25
  codedoc_full     code         16   1,200,000    39.05   38.21%     6/25
```

**The corpus change is CONFIRMED on all three measures:**
* perplexity **44.26 -> 39.05** (better)
* next-token accuracy **36.44% -> 38.21%** (better)
* executable syntax **4/25 -> 6/25**, i.e. **16% -> 24%**

And the prediction held: *improved, and still low.* Removed prose makes a better language model over
code, and it makes a little more of the output runnable -- but 24% is nowhere near usable, so nothing
here is claimed as working code generation.

### WHY IT STILL FAILS, AND THE NEXT CHANGE IS ALREADY SIZED

`<unk>` remains the most common token in the generated output. The output is still full of
`def <unk> ( <unk> , <unk> )`, because the model has no tokens to name anything with:

```
  vocab    <unk> rate          on a REAL 322-token solution:
   6,000      7.18%              at 6k  -> 26 tokens (8.1%) unknown
  32,000      3.51%              at 32k ->  1 token  (0.3%) unknown
  50,000      2.47%              (and even 50k leaves 2.47%: identifiers are unbounded)
```

Change 2 -- vocabulary 6k -> 32k, everything else identical -- is running now. The honest caveat stays
attached: **32k is a large improvement, not a solution.** With 136,030 distinct tokens in only 6.2M
tokens, a compositional scheme that builds identifiers from sub-tokens would be needed for the rest,
and that is not built.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
for tag, s in (("code1  (both, 6k)", c1), ("codedoc (UNFAIR 4ep)", cd), ("codedoc_full (code, 6k)", cf)):
    if s:
        print(f"  {tag:<26} ppl {s['ppl']:>7.2f}  acc {s['acc']:>6.2f}%  "
              f"{s['n']} epochs  params {s['params']:,}")
print(f"\n  executable test: syntax {rx['syntax']}/{rx['total']}, runs {rx['ran']}/{rx['total']} "
      f"({rx['ran']/rx['total']*100:.0f}%)")
print("=" * 100)
