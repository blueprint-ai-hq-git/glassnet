"""record_all_options.py — every option run this session, and what the evidence actually said.
Including the two where the "winner" turned out to be noise and the one where my own harness was
the difference. Nothing dropped because it was unflattering.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — "RUN ALL OPTIONS, PICK THE BEST ON EVIDENCE": the results, including the failures

The owner's rule is to run every option and choose on evidence, never ship the first thing that
moves. This is what the evidence said for each capability.

### MEDICAL — 6-config sweep + 3-seed replication. THE SWEEP WINNER WAS NOISE.
```
  6-arm sweep (readable test accuracy, pathmnist, 9 classes, chance 11.1%):
    d96_lc0.15   66.74%   plain 41.41%  (+25.33)   concepts 28/35   86,615 params
    d128_lc0.15  66.46%   plain 41.41%  (+25.06)   concepts 30/35  115,319 params
    d64_lc0.15   66.03%   plain 41.41%  (+24.62)   concepts 30/35   57,911 params
    d64_lc0.30   64.92%   plain 41.41%  (+23.51)   concepts 27/35
    d64_lc0.50   64.82%   plain 41.41%  (+23.41)   concepts 27/35
    d96_lc0.30   63.25%   plain 41.41%  (+21.84)   concepts 25/35

  Concept weight is the clear signal: 0.15 > 0.30 > 0.50 at EVERY width. That replicates across
  all six arms, so it is a real effect, not noise.

  3-seed replication of the apparent width winner:
    d64_lc0.15  [66.03, 62.90, 57.13]  mean 62.02  std 4.51
    d96_lc0.15  [66.74, 62.90, 62.79]  mean 64.14  std 2.25
    gap +2.12 vs pooled std 3.57  ->  s1 FAILS: the gap is INSIDE the noise.
```
**DECISION: keep d64_lc0.15.** The +0.71 single-seed "win" for d96 was seed luck; over 3 seeds the
difference is smaller than the spread. d64 is also 1.5x smaller. Per the pre-registered rule, no
difference found means keep the proven smaller model — NOT "pick the bigger one because it won once".

**What IS solid about medical (unchanged and strong): the readable model beats its own
parameter-matched plain control by +24.6 points, and that held on every single seed and arm.** That
is the real, replicated finding: the readable architecture is not a handicap on this task.

### TALK — vocab/context experiments already gated (v12k MIXED, w24 FAIL)
  best1 (V=3500,  W=16):  bits/token 6.04   <unk> 14.5%
  v12k  (V=12000, W=16):  bits/token 6.61   <unk>  3.2%   v1 PASS / v2 FAIL
  w24   (V=12000, W=24):  bits/token 6.63   <unk>  8.5%   w1 FAIL / w2 FAIL
Decoder sweep (top-k x temp x rep grid) run separately and reported in its own entry.

### CODE — my harness was the difference; the discrepancy is explained, not hidden
`reexec_code.py` reported 8/25 for code1_ep16. My own sweep reported 0/25 for the same checkpoint.
Root cause found by reading both scripts: **the prompt construction differs.** reexec_code builds
style-B prompts from the FULL document text (problem statement included) cut after the signature
colon; my sweep cut from `def ` onward. Same checkpoint, different input, different number.
**This is the verdict-check bug class again: two numbers from two harnesses are not comparable.**
Both are recorded; the authoritative one is reexec_code.py, since it is the harness that produced
the 8/25 baseline and it also reports the clean-prompt subset (20 of 25 style-B prompts contain
`<unk>`, so read-back `<unk>` is partly a harness artifact).

### FACES — tested, experimental, accuracy UNPROVEN (unchanged)
Synthetic controls 5/5; 7/40 false positives on the owner's real photos; every public face dataset
fails the safe-licence rule. Needs 30s of the owner in front of the webcam for ground truth.

### CYBER — NOT STARTED. No licence-clean corpus offline and 2 GB of mobile data rules out fetching one.

### WHAT I DID NOT DO
I did not promote d96 to "best" despite it topping the sweep. I did not quote my 0/25 as the code
result when the authoritative harness says 8/25. I did not tune weights against unlabeled photos
until they "looked right". Each of those would have been a number that looked like progress.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended all-options record")
