"""record_concepts_test.py — the concept layer verdict: width helps, READABILITY did not (c1 fail).
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "concepts", "concepts_more.json")))
r = d["res"]

LOG = f"""
---

## 2026-10-08 — CONCEPT CEILING TEST: the band breaks at +4 pts, but NOT because of understanding

The experiment three findings pointed to. Same data/budget, three arms, 3 seeds:
```
  base  (12 named concepts)      {r['base']['mean']:.2f}% ± {r['base']['std']:.2f}   K=12
  real  (12 + 10 NEW named)      {r['real']['mean']:.2f}% ± {r['real']['std']:.2f}   K=22-ish
  dummy (12 + 10 RANDOM pixel projections) {r['dummy']['mean']:.2f}% ± {r['dummy']['std']:.2f}  K=21
```
```
  c1 real beats dummy >2:    {r['real']['mean']-r['dummy']['mean']:+.2f}  FAIL
  c2 real beats base  >2:    {r['real']['mean']-r['base']['mean']:+.2f}   PASS
  c3 dummy ~= base (within 2): {r['dummy']['mean']-r['base']['mean']:+.2f}  FAIL (dummy beat base too)
```

### THE VERDICT, WRITTEN BEFORE ANY SPIN

**The concept layer WAS a bottleneck (c2: more concepts broke 74 -> 78.5) — but the unlock is
WIDTH, not SEMANTICS: ten RANDOM linear pixel projections gained nearly as much (+3.6) as ten
hand-written readable concepts (+4.4), and real-vs-dummy (+0.77) is inside seed noise.**

What that honestly means for GlassNet's thesis:
* On THIS task (10 clothing classes, 784 px, next-word), named concepts at K=12 are *information-
  starved*, not wrong: the answer needed more directions in the bottleneck, readable or not.
* "Transparency costs nothing" survives (real >= dummy, so readable concepts are not hurting).
* "Transparency itself improves learning" is **NOT SUPPORTED at this rung** and we do not claim
  it. The readable-concept value demonstrated elsewhere today is different and still real: exact
  explanations (residual 0.00e+00), audited firing (AUROC), refusal behaviour. Understanding the
  model ≠ teaching the model, and this test says the second part did not happen here.
* The 78.5 ceiling is the NEW band (both 22-concept arms ~78), matching the p1-miss and eyes null:
  on this task family everything points at the same wall — 10-way class info in 784 px. The
  remaining untried lever at this rung is K again (32/64), but the pattern says gains will flatten;
  the honest next move with expected upside is NEW INFORMATION, not more of the same: licence-clean
  open-vocab captions or the (b)/(c) face decision, both of which are owner decisions.

### BUGS CAUGHT IN THE EXPERIMENT DESIGN ITSELF (pre-run, by re-reading)
* noun mask compared FULL names ("ankle_boot") against vocab WORDS ("ankle") — would have matched
  zero rows and silently scored nothing; fixed to token sets, with a hard stop if the mask empties.
* one spec (wide_object) had an axis error AND a tautology (`& True`) — replaced by tall_object.

Artifacts: concepts_more.json (raw per-seed accuracies, gates, verdict string).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
print("=" * 92)
print(f"  base {r['base']['mean']:.2f}  real {r['real']['mean']:.2f}  dummy {r['dummy']['mean']:.2f}")
print("  c2 PASS (K helps, +4.4) | c1+c3 FAIL -> WIDTH not READABILITY unlocked it; said plainly")
print("=" * 92)
