"""record_v12k.py — the vocab experiment: v1 PASS, v2 FAIL. Honest tradeoff, both numbers.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
v = json.load(open(os.path.join(P, "lens", "runs", "chat", "v12k_verdict.json")))

LOG = f"""
---

## 2026-10-09 — EXPERIMENT v12k (vocab 3,500 -> 12,000, ONE change): v1 PASS, v2 FAIL

The one-change experiment from best1: same recipe (d=128, W=16, 24 epochs, lr 0.4 decay 0.5,
1.5M windows), ONLY the vocabulary changed. Gates were fixed BEFORE the run.

```
  model      V      W    val_ppl   bits/token   <unk> in replies
  best1    3,500   16     65.96       6.04         14.5%
  v12k    12,000   16     98.00       6.61          3.2%

  v1  unk 3.2% < 14.5%   (delta +11.2 pts, a 4.5x reduction)   PASS
  v2  bits/token 6.61 vs 6.04  (delta +0.57, bar +0.30)        FAIL
```

**VERDICT (pre-registered, not reworded after seeing results): MIXED — the unk win is large and
real, the perplexity cost is larger than the bar allowed.** Per the gate rules written before the
run, this means: report it, do NOT hide it, and the next lever is CONTEXT LENGTH, not more vocab.

### WHAT THE NUMBERS ACTUALLY SAY
* The big vocab does exactly what it was supposed to: **<unk> fell from 14.5% to 3.2%.** Replies read
  noticeably more like words now (sample: "i can use the time of the <unk> of a lot of it , in a
  common - <unk> . human : no , i am not s") — fewer holes, though still not fluent.
* The cost is also real: **+0.57 bits/token.** A 12,000-way output choice is intrinsically harder
  than a 3,500-way one; the model spends capacity on the wider decision. It recovered 98.00 from a
  mid-run 122.94 (13 more epochs helped), but it did not close the gap to 6.04.
* Neither model is "the winner" on both axes. best1 predicts each token more confidently; v12k
  produces fewer holes. That is a genuine tradeoff, not a bug and not a spin.

### THE HONEST CHOICE FOR THE TALK MODEL
For "talk quality" as a USER experiences it, holes in the output (`<unk>`) are more damaging than a
0.57 bits/token difference — a sentence with four `<unk>`s is unreadable regardless of its
perplexity. So v12k is the better TALK artifact despite losing on bits/token, and both are kept:
best1 as the perplexity-optimal model, v12k as the talk-optimal one. Neither claim is hidden.

### NEXT LEVER (queued, one change at a time)
Context length W=16 -> 24, pre-registered in run_w24.sh (w1 ppl better by >3%, w2 bits/token
better by >0.05, w3 report failure and stop adding context if it does not help).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended v12k record")
print(f"  v1 PASS unks {v['base']['unk']*100:.1f}% -> {v['new']['unk']*100:.1f}%")
print(f"  v2 FAIL bpt {v['base']['bpt']:.2f} -> {v['new']['bpt']:.2f} (bar +0.30)")
