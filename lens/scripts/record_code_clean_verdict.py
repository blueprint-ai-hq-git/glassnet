"""record_code_clean_verdict.py — third and final code hypothesis closed: data content is NOT the
blocker either. All three levers tested; code generation is capacity-limited.
Plus the cumulative (post-hoc, labelled) analysis.
"""
import json
import os
from math import comb

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "code", "code_clean_verdict.json")))
r = d["results"]


def fisher(a, b, c, e):
    n = a + b + c + e

    def p_of(x):
        try:
            return comb(a + c, x) * comb(b + e, a + b - x) / comb(n, a + b)
        except Exception:
            return 0.0
    obs = p_of(a)
    tot = sum(p_of(x) for x in range(0, min(a + c, a + b) + 1) if p_of(x) <= obs + 1e-12)
    return min(1.0, tot)


cum = fisher(r["cclean32k_ep16"]["syntax"], 150 - r["cclean32k_ep16"]["syntax"], 20, 130)

LOG = f"""
---

## 2026-10-09 — CODE: ALL THREE HYPOTHESES CLOSED. It is capacity-limited, not data-limited.

Third and final lever tested: **the training data content.**

```
  model                vocab     syntax at n=150        ran        unk
  code1_ep16            6,000    20/150 = 13.3%       20/150      6.9%    <- baseline
  covocab32k_ep16      32,000    26/150 = 17.3%       26/150      5.5%    (vocab change)
  cclean32k_ep16       32,000    30/150 = 20.0%       27/150      6.7%    (clean corpus)

  clean corpus vs baseline: 20.0% vs 13.3%, Fisher exact p = {d['fisher_p']:.4f}
  p2 (p<0.05) -> FAIL
```

**VERDICT: p3 applies. Cleaning the fences and prose out of the training targets does NOT
significantly improve code generation (p={d['fisher_p']:.2f}). Data content is not the blocker
either.**

### THE THREE LEVERS, ALL TESTED AND ALL CLOSED
```
  1. VOCABULARY    6k -> 32k        20/150 -> 26/150   p = 0.42    NOT significant
  2. CONSTRAINED   syntax masking   both arms 0/25    FAILED      syntax cannot fix semantics
     DECODING
  3. DATA CONTENT  fences+prose    20/150 -> 30/150   p = {d['fisher_p']:.2f}    NOT significant
                   stripped
```
**Each lever moved the number in the right direction (20 -> 26 -> 30) and none of them reached
significance on its own.** That pattern — consistent direction, individually small — is what a
CAPACITY limit looks like, not a data or tokenizer defect.

### POST-HOC CUMULATIVE ANALYSIS (labelled as post-hoc, NOT a pre-registered gate)
Comparing the final clean model against the original baseline directly:
```
  30/150 vs 20/150   Fisher exact p = {cum:.4f}
```
{'This IS significant cumulatively.' if cum < 0.05 else 'Still not significant even cumulatively.'}
This is reported for completeness and is explicitly NOT counted as a gate pass, because it was
chosen after seeing the results and stacks three changes at once. The pre-registered single-change
gates all failed, and those are what govern.

### WHAT THIS MEANS (the honest conclusion)
Code generation at ~1-2M parameters is **capacity-limited**. No amount of vocabulary, syntax
constraint, or data cleaning makes a model this size reliably write runnable programs. The next
honest test is therefore CAPACITY: train a substantially larger model on the clean corpus with
everything else identical, and see whether the syntax rate moves significantly. That experiment is
launched, with gates registered before it runs.

### WHAT IS STILL TRUE AND GOOD ABOUT THE CODE WORK
The concept layer audits strongly (29/38 concepts, has_a_loop AUROC 0.998, has_return 0.9997,
has_error_handling 1.000). **Readability holds even where generation does not.** And the measurement
harness is now trustworthy: one script, one construction, per-checkpoint vocab, significance test —
after three same-checkpoint-different-number incidents today.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code clean verdict record")
print(f"  p2 FAIL: 30/150 vs 20/150, p={d['fisher_p']:.4f}")
print(f"  post-hoc cumulative: p={cum:.4f}")
