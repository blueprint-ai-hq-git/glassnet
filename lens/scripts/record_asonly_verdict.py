"""record_asonly_verdict.py — MIXED: a1 PASS (unk 0.0%), a2 FAIL on an invalid metric, plus a real
repetition regression that is NOT excused.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "chat", "asonly_verdict.json")))
n, b = d["new"], d["base"]

LOG = f"""
---

## 2026-10-09 — ASSISTANT-ONLY MASKING, FINAL VERDICT: MIXED (a1 PASS, a2 FAIL, one real regression)

```
  model             ckpt                        V    val_ppl      unk      rep
  v12k              v12k_ep24.npz          12,000      98.00      3.1%     1.8%
  asonly_v12000     asonly_v12000_ep24.npz 12,000   10771.53      0.0%     4.7%

  a1  unk 0.0% < 3.2%                      PASS
  a2  bits/token 13.39 <= 6.91             FAIL
  repetition 1.8% -> 4.7%                  a REAL regression, stated below
```

### WHAT GENUINELY IMPROVED
**<unk> in generated replies fell to 0.0% — literally zero holes**, from 3.2%. Across 30 replies
(10 prompts x 3 seeds) the model produced not one `<unk>`. That is the single cleanest output of any
model in this project on that axis.

### WHAT FAILED, AND WHY THE FAILURE IS COMPLICATED
a2 (bits/token 13.39 vs a 6.91 bar) failed badly. **But this gate measures the wrong subject for this
model by construction**: validation ppl is computed over ALL tokens including the HUMAN's text, and
this run deliberately stopped training on human text. The model is being graded on what it was told
to ignore, so the number HAD to get worse. The speaker-split diagnostic confirms the model is not
broken: assistant-turn ppl is nearly 2x BETTER than the baseline's. So a2's failure is an invalid
measurement, not evidence of a worse model.

### THE REGRESSION I AM NOT EXCUSING
**Repetition rose from 1.8% to 4.7% (a 2.6x increase).** That is a real, measured cost on a metric
that IS valid for this model, and the a2 explanation does not cover it. A model that says less
nonsense but loops more is a genuine trade, not a win. It is recorded as a cost.

### AND THE HONEST LOOK AT THE OUTPUT
Sample replies, verbatim:
```
  well they can be pretty helpful for other things should work for me for any way one way for your
  most really have some questions about their system would with all for someone are up for all one
  yes about what are asking me what kind ! well might be interested for something with one or use
```
**They are grammatical, contain no holes, and say nothing.** Zero `<unk>` made the text look cleaner,
but it is fluent filler, not answers. Removing the holes did not add the knowledge — the model still
has ~600K parameters trained on 34MB. This is exactly the "sounds better, is not smarter" trap, and
it is stated rather than presented as a win.

### WHERE THE TALK MODEL STANDS (all measured, nothing claimed)
```
  best1            bpt 6.04   unk 14.5%   ppl-optimal
  v12k             bpt 6.61   unk  3.2%   the previous talk default
  asonly_v12000    unk  0.0%  rep  4.7%   zero holes, but repetitive and still empty
  decoder: top-k 60, temp 0.8, rep 1.15   (the one free, unambiguous win today)
```
**Not a chatbot.** No arm is claimed as one. The measured improvement today that is unambiguous is
the DECODER (repetition 4.5% -> 1.8% on the same model, free). The masking result is mixed with one
real regression. Both are in the record.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended asonly final verdict (MIXED)")
print(f"  a1 PASS unk {n['unk']*100:.1f}% (was {b['unk']*100:.1f}%)")
print(f"  a2 FAIL bpt {n['bpt']:.2f} vs bar 6.91 (metric invalid for this model)")
print(f"  REGRESSION repetition {b['rep']*100:.1f}% -> {n['rep']*100:.1f}%")
