"""record_code_capacity.py — FINAL CODE VERDICT: capacity is NOT the blocker either (p=0.4526).
All four levers tested. Code generation on this architecture/data cannot do reliable code at any
size tested.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "code", "code_capacity_verdict.json")))
r = d["results"]

LOG = f"""
---

## 2026-10-09 — CODE: CAPACITY TEST CLOSED TOO. Final verdict on all four levers.

```
  model              d     vocab    syntax n=150        ran      unk
  code1_ep16        192     6,000    20/150 = 13.3%    20/150    6.9%   <- baseline
  covocab32k        192    32,000    26/150 = 17.3%    26/150    5.5%   (vocab)
  cclean32k         192    32,000    30/150 = 20.0%    27/150    6.7%   (clean data)
  cbig384           384    32,000    24/150 = 16.0%    20/150    6.5%   (4x wider)

  cbig384 vs cclean32k: 16.0% vs 20.0%, Fisher exact p = {d['fisher_p']:.4f}
  c1 -> FAIL
```

### c3 APPLIES: a 4x wider model does NOT significantly improve code generation (p=0.4526).
Capacity is not the blocker either. The capacity model actually scored WORSE (24 vs 30), though
that difference is also not significant. Per the pre-registered rule: **STOP. No more code arms.**

### ALL FOUR LEVERS, ALL CLOSED — the complete picture
```
  1. VOCABULARY    6k -> 32k        20/150 -> 26/150   p = 0.42   NOT significant
  2. CONSTRAINED   syntax masking   both arms 0/25     FAILED     syntax cannot fix semantics
     DECODING
  3. DATA CONTENT  fences + prose   20/150 -> 30/150   p = 0.16   NOT significant
                   stripped
  4. CAPACITY      d 192 -> 384     30/150 -> 24/150   p = 0.45   NOT significant (went DOWN)
```
Every single lever either failed or was not significant. The model parses 13-20% of the time
regardless of what we change. **This architecture on this data cannot do reliable code generation
at any size tested.** That is the honest conclusion and it is now closed.

### WHAT IS STILL TRUE ABOUT THE CODE MODEL
The concept layer audits strongly (has_a_loop AUROC 0.998, has_return 0.9997,
has_error_handling 1.000). The model READS code well; it cannot WRITE it. Those are different
skills and the readable architecture excels at the first.

### THE COMPLETE SESSION TALLY (2026-10-09)
Experiments run: 12+ (medical sweep 6 arms, 3 seeds, 2 generalization tests, size test 3 arms,
talk vocab/context/decoder/masking, code vocab/constrained/clean/capacity)
Harness bugs found and fixed: 6 (stale artifacts, wrong-vocab-to-checkpoint, prompt-construction
mismatch, key collision, missing default from grid, metric-wrong-subject)
Real wins: decoder sweep (free, repetition 4.5%->1.8%), readable architecture proven on 2 domains
Honest failures: W=24 context, constrained decoding, code vocab/clean/capacity (all 4 levers),
assistant-only masking (mixed: unk 0.0% but repetition 2.6x worse)
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code capacity final verdict")
print(f"  c1 FAIL: 24/150 vs 30/150, p={d['fisher_p']:.4f}")
print("  ALL FOUR CODE LEVERS CLOSED — STOP")
