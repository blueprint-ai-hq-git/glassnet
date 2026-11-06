"""record_talk_fair.py — fair comparison: every talk model at its OWN best decoder.
asonly wins on BOTH metrics. u1 PASS. This is the defensible talk result.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "chat", "talk_fair_comparison.json")))
m = d["models"]
a, v, b1 = m["asonly_v12000"]["best"], m["v12k"]["best"], m["best1"]["best"]

LOG = f"""
---

## 2026-10-09 — TALK: FAIR COMPARISON. asonly wins on BOTH metrics. This is the defensible result.

### WHY A FAIR COMPARISON WAS NEEDED
I tuned a decoder for the assistant-only model and found it beat v12k's levels — but v12k and best1
were still being judged at INHERITED settings. Comparing "asonly at its best" against "best1 at its
default" is the same unfair comparison caught three times today. **So every model got its own decoder
sweep (36 settings x 3 seeds) and only then were they ranked.**

```
  model              its best decoder        unk       repetition
  asonly_v12000      k{a['top_k']} t{a['temp']} r{a['rep']}          {a['unk_rate']*100:5.2f}%      {a['rep_rate']*100:5.2f}%
  v12k               k{v['top_k']} t{v['temp']} r{v['rep']}          {v['unk_rate']*100:5.2f}%      {v['rep_rate']*100:5.2f}%
  best1              k{b1['top_k']} t{b1['temp']} r{b1['rep']}          {b1['unk_rate']*100:5.2f}%      {b1['rep_rate']*100:5.2f}%

  u1  winner <= runner-up on BOTH metrics -> PASS (asonly wins both, no trade-off)
```

### VERDICT
**asonly_v12000 is the best talk model on a fair comparison: unk {a['unk_rate']*100:.2f}% and
repetition {a['rep_rate']*100:.2f}%.** It wins unk outright (zero holes vs 2.58% and 6.08%) and also
has the lowest repetition — so this is not a trade-off dressed up as a win. The advantage survived
the fair test, which is what makes it credible: it did not win merely because it got a tuned decoder.

### WHAT CHANGED ACROSS THE DAY FOR TALK (the full arc, nothing dropped)
```
  stage                                     unk      repetition   verdict
  best1 (V=3500)                           14.5%      n/a         ppl-optimal, hole-heavy
  v12k (V=12000)                            3.2%      n/a         v1 PASS / v2 FAIL -> MIXED
  v12k + top-k 60 decoder                   3.1%      1.8%        decoder WIN (free)
  w24 (window 24)                           8.5%      n/a         both gates FAIL -> closed
  asonly + inherited decoder                0.0%      4.7%        MIXED (regression)
  asonly + tuned decoder (k{a['top_k']} t{a['temp']} r{a['rep']})     {a['unk_rate']*100:.2f}%      {a['rep_rate']*100:.2f}%        WIN, and it survives a FAIR test
```

### STILL TRUE AND STILL SAID PLAINLY
The replies are grammatical, hole-free, minimally repetitive — and still largely EMPTY. "well they
can be pretty helpful for other things" is fluent filler, not an answer. **This is a reply-QUALITY
win on measurable axes, not a chatbot claim.** ~600K parameters on 34MB of chat data cannot answer
questions it has no knowledge of, and no decoder change alters that.

### RULE ADDED (the third time today this class appeared)
When a model changes, its decoder is stale. And when comparing models, EVERY model must be measured
at its own best settings — otherwise the comparison measures tuning effort, not model quality.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended talk fair comparison")
print(f"  asonly k{a['top_k']} t{a['temp']} r{a['rep']}: unk {a['unk_rate']*100:.2f}% rep {a['rep_rate']*100:.2f}%")
print(f"  v12k {v['unk_rate']*100:.2f}%/{v['rep_rate']*100:.2f}%  best1 {b1['unk_rate']*100:.2f}%/{b1['rep_rate']*100:.2f}%")
