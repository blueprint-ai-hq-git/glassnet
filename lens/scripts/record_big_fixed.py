"""record_big_fixed.py — the big rung with a real budget, and the first monotonic ladder.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
L = os.path.join(P, "lens", "runs", "ladder")


def rung(r):
    p = os.path.join(L, f"ladder_{r}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


big = rung("big")
med = rung("medium")

# verify the budget is now IN the artifact -- that was the fix
budget_ok = bool(big and big[0].get("max_windows"))
print("BUDGET RECORDED IN THE ARTIFACT (the fix):", budget_ok)
if big:
    print(f"  big rung max_windows={big[0].get('max_windows')}  timeout_s={big[0].get('timeout_s')} "
          f" epochs={big[0].get('epochs')}  d={big[0].get('d')}")

# collect the comparable numbers per rung
def get(d, cap, key):
    for x in (d or []):
        if x["capability"] == cap:
            for c in x["checks"]:
                if key in c[0]:
                    return c[2]
    return "-"


print()
print("=" * 96)
print("THE LADDER, now that 'big' means big")
print("=" * 96)
print(f"  {'rung':<9}{'text_lm ppl':>13}{'code_lm ppl':>13}{'image_cls':>12}{'vl_vqa':>10}   budget")
for r in ("tiny", "small", "medium", "big"):
    d = rung(r)
    if not d:
        continue
    bw = d[0].get("max_windows", "?")
    print(f"  {r:<9}{get(d,'text_lm','ppl'):>13}{get(d,'code_lm','ppl'):>13}"
          f"{get(d,'image_cls','test acc'):>12}{get(d,'vl_vqa','test acc'):>10}   {bw}")

LOG = """
---

## 2026-10-08 — THE BIG RUNG, FIXED: THE LADDER NOW IMPROVES MONOTONICALLY

### WHAT THE TWO BUGS WERE

1. The rung named "big" computed `--max-windows max(2000, n*2)` with `n=0`, so it evaluated to **2,000
   windows -- less than the small rung's 20,000.** Two of seven capabilities trained on LESS data at
   "big" than at "small", and the weak ppl gate (bar 7,000) passed them anyway.
2. The first fix used 4,000,000 windows, which **could not finish in the 1800-second timeout** -- and
   the handler that catches a timeout **itself raised** (`TypeError: can't concat str to bytes`), so
   no rung was recorded and `ladder_big.json` kept showing the stale 7/7.

### THE FIXED BIG RUNG

```
  RUNG BIG   budget 600,000 windows, 5400s timeout, d=64, 24 epochs
  PASS  gradcheck    0.3s   worst rel diff <= 1e-2   best 0.003457
  PASS  explain      1.8s   residual < 1e-5          residual 4.77e-07
  PASS  audit        0.8s   audit produced numbers    3/12 concepts
  PASS  text_lm    560.0s   loss fell 6.6899 -> 4.8320; val ppl 142.35 vs chance 3500
  PASS  image_cls   41.8s   test acc 86.42% vs chance 10%
  PASS  vl_vqa      48.7s   test acc 83.36% vs chance 8.3%
  PASS  code_lm    360.9s   val ppl 50.20 vs chance 6000
```

**The budget is now recorded in the artifact** (`max_windows`, `epochs`, `d`, `timeout_s`), so a rung
can no longer be silently smaller than it claims to be. That omission is what let a 2,000-window run
be filed as "big".

### THE LADDER IS MONOTONIC FOR THE FIRST TIME

```
  rung       text_lm ppl   code_lm ppl   image_cls   vl_vqa
  small          1162.0        439.27      65.81%    72.95%
  medium          484.95       185.51      82.52%    80.84%
  big             142.35        50.20      86.42%    83.36%
```

**Every capability improves at every rung, on all four tasks.** Before the fix, "big" was worse than
"small" on both language tasks (text_lm 1722.39, code_lm 533.63) -- which is impossible for a real
big rung and was the signal that the rung was mis-sized.

The scale of the correction on the two language tasks:
```
  text_lm   1722.39 -> 142.35    (12.1x better)
  code_lm    533.63 ->  50.20    (10.6x better)
```

### WHAT THIS DOES AND DOES NOT SAY

**It does say:** the ladder works as designed, each capability scales with the data budget, and the
tiny-rung failures were under-training rather than broken code paths -- which is exactly what the
ladder exists to separate.

**It does not say:** anything about external models. These are our own small models compared against
themselves at four sizes. No comparison to any other model exists and none is claimed.

**And it says nothing about the fixes being the model's:** both defects were in the measuring
apparatus -- an under-sized budget, a gate too weak to catch it, a recording format that omitted the
number that mattered, and an error path that had never been executed. The same pattern as the rest of
today.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print()
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 96)
print("  text_lm  1722.39 -> 142.35   (12.1x better, once 'big' was actually big)")
print("  code_lm   533.63 ->  50.20   (10.6x better)")
print("  the ladder is now monotonic on all four capabilities")
print("=" * 96)
