"""record_capability_audit.py — honest state of the three capabilities the owner asked for:
talk, coding, medical. Real numbers, including the ones that are bad.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — CAPABILITY AUDIT (talk / coding / medical): what is REAL and what is BROKEN

The owner asked for one assistant good at talking, coding, and medical/cyber. Before claiming
anything, I measured all three from the artifacts on disk. One is strong, one is a broken number
that LOOKED good, and one is mid-experiment.

### 1. MEDICAL — GENUINELY STRONG (the best result this project has)
PathMNIST histopathology, 9 tissue classes (adipose, cancer_associated_stroma, colorectal
adenocarcinoma, ...), chance = 11.1%. Readable model vs parameter-matched plain control:

```
  run            readable     plain      delta      concepts passing audit
  med_cc          66.03%     41.41%     +24.62       30/35
  med_cc_s1       62.90%     41.41%     +21.49       29/35
  med_adapt       52.87%     41.41%     +11.46        8/20
  med_bal          8.68%      4.68%      +4.00        0/16   <- FAILED run, kept on disk
```

**+24.6 points over the plain control at 66.03% on real medical images, with 30/35 named concepts
passing their own firing test.** This is the strongest evidence in the whole project that the
readable architecture is not a handicap — on this task it WINS by a wide margin. (med_bal is a
failed balancing experiment, kept and labelled rather than deleted; med_cc beat it clearly.)

### 2. CODING — THE NUMBER WAS LYING AND I AM FLAGGING IT
The code model reports val ppl **39.05** (best of any model in the project — better than the chat
model's 65.96). That looks like the strongest model here. It is not:

```
  codedoc_full:  ppl 39.05,  25 code samples executed -> syntax_ok 0/25, ran 0/25
  code1:         ppl 44.26,  25 code samples executed -> syntax_ok 0/25, ran 0/25
  codedoc:       ppl 91.06,   8 code samples           -> syntax_ok 0/8
```

**Zero of the generated programs even PARSE.** A low perplexity on token prediction does not mean
runnable code, and this is the clearest example of it in the project. The model learned the
statistics of code text without learning to close brackets, indent, or produce a complete
statement. What IS real: the concept layer audits well (29/38 concepts pass, including has_a_loop
AUROC 0.998, has_return 0.9997, has_error_handling 1.000) — so the READABILITY claim holds even
where the GENERATION claim does not.

**Honest status: coding is NOT working.** ppl must never be quoted for this model again without the
0/25 exec rate beside it. Next real test would be constrained decoding (only emit tokens that keep
the program syntactically valid), which is a different mechanism, not a bigger model.

### 3. TALK — MID-EXPERIMENT, one gate passed one failed
```
  best1 (V=3500,  W=16):  bits/token 6.04   <unk> 14.5%
  v12k  (V=12000, W=16):  bits/token 6.61   <unk>  3.2%   v1 PASS / v2 FAIL
  w24   (V=12000, W=24):  RUNNING NOW (pre-registered w1/w2/w3)
```
The unk reduction (14.5% -> 3.2%) is real and large; the perplexity cost (+0.57 bpt) exceeded the
pre-registered bar, so it is recorded as MIXED, not as a win.

### 4. CYBER SECURITY — NOT STARTED, and honestly it needs the same treatment
No cyber model exists on disk. Under the standing rule (safe licences only) the same licence work
that blocked faces applies. Not claimed, not started.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended capability audit")
print("  MEDICAL: med_cc readable 66.03% vs plain 41.41% = +24.62 pts, 30/35 concepts pass")
print("  CODING:  ppl 39.05 LOOKS best but 0/25 samples parse — flagged, not working")
print("  TALK:    v12k unks 14.5%->3.2% PASS, bits/token +0.57 FAIL (mixed)")
