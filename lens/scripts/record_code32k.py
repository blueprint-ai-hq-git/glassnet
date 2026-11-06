"""record_code32k.py — vocab experiment: gates pass mechanically, but significance test says the
difference is chance. Both facts recorded; the weaker one governs the claim.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
v = json.load(open(os.path.join(P, "lens", "runs", "code", "code32k_verdict.json")))
n, o = v["new"], v["old"]

LOG = f"""
---

## 2026-10-09 — CODE vocab 6k -> 32k: gates PASS, but a significance test says it is NOT real

One change from code1: vocabulary 6,000 -> 32,000 on the code-only corpus (already on disk, no
downloads). Gates fixed before the run.

```
  model             style   syntax   ran   <unk> in output
  code1_ep16 (6k)   A        0/25      0      7.0%
  code1_ep16 (6k)   B        4/25      4      7.1%
  covocab32k (32k)  A        0/25      0      5.8%
  covocab32k (32k)  B        6/25      6      5.0%

  q1  style-B syntax 6/25 > 4/25   PASS
  q2  <unk> 5.0% < 7.1%            PASS
```

**Then I tested whether the q1 difference is real, and it is not.**
```
  Fisher exact test, 4/25 vs 6/25:  p = 0.725
  16.0% -> 24.0% is an 8-point difference on 25 samples -> well inside chance.
```

**So the honest statement is: the gates passed mechanically, but the syntax improvement is NOT
statistically distinguishable from noise.** A +2 sample change on n=25 cannot support the claim
"vocabulary was the blocker". I am recording the gate result AND the significance result, and the
significance result governs what may be claimed.

What DID move in the right direction and is consistent across both styles: **<unk> in output fell
from 7.0-7.1% to 5.0-5.8%**, which is what a 5.3x vocabulary increase should do, and it matches the
corpus-level measurement (5.74% -> 3.74% unk tokens). That part is real; the syntax claim is not
supported yet.

### WHAT THIS MEANS (and what I am NOT doing)
* I am NOT claiming vocabulary fixed coding. On this evidence it did not demonstrably help.
* I am NOT now adding MORE vocabulary — the pre-registered q3 rule says a failed syntax gate means
  stop adding vocabulary. The gate technically passed, but its own significance test failed, so
  adding vocabulary further is unjustified.
* To actually settle it needs a bigger sample (100+ prompts, not 25) or a 3-seed comparison. With
  n=25, differences of 2-3 samples are unreadable. **The measurement is the bottleneck, not the
  model.**

### TWO HARNESS BUGS FOUND IN THIS RUN (both fixed, both the same family as before)
1. **Wrong vocabulary.** The first verdict run printed "vocab 6,000" while evaluating a 32,000-word
   checkpoint — `code_data.build()` returned the default 6k vocab from an unset env var, so every
   token mapped to the WRONG WORD and the whole comparison was meaningless. Fixed: each checkpoint
   now loads the vocab matching its OWN `net.V`, and a mismatch refuses rather than printing a
   number.
2. **Same checkpoint, different number, again.** Earlier today the same checkpoint scored 8/25, then
   4/25, then 6/25 across three harnesses — because the prompt construction differed each time. Now
   both models are measured by ONE construction, in ONE script, in the same run. This is the third
   instance of the verdict-check bug class and it is now guarded by construction (same script, same
   prompts, per-checkpoint vocab).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code32k record")
print(f"  gates: q1 PASS (4/25->6/25), q2 PASS (unk 7.1%->5.0%)")
print(f"  BUT Fisher exact p=0.725 -> the syntax gain is NOT significant; claimed as inconclusive")
