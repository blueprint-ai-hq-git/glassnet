"""record_sweep.py — the trade-off curve, and what it implies.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 15:25 — THE SWEEP: A CLEAN TRADE-OFF CURVE, AND NO WIN.

```
lam_concept    concepts       ppl       gap      acc
0.1                 7/9    90.085   +12.281   29.73%
0.3                 9/9   100.850   +23.045   29.62%
0.5                 9/9   107.020   +29.216   29.65%
1.0                 9/9   115.829   +38.025   29.65%
matched normal        -    77.804         -    28.43%
```

**Pre-registered before the sweep finished:** "I expect 9/9 to be reached somewhere between 0.3 and
0.5, and the gap to scale roughly with `lam_concept`. If a setting gives 9/9 at a gap under +10,
that is a genuine win on the blocker."

**Result:**
* 9/9 is reached at **0.3**, slightly earlier than predicted.
* The gap scales with `lam_concept` -- **predicted correctly**, and it is monotonic across all four
  points.
* **No setting gives 9/9 under +10. The cheapest is +23.045 against a bar of ~6. NOT A WIN.**

**What the curve means.** The cost rises smoothly and monotonically with concept supervision
strength. That is the signature of a cost that lives in the **supervision itself**, not in a
particular setting -- so no amount of tuning removes it. Target I's sparsity penalty was a dead
end; this confirms the direction of the search was wrong.

**The other consistent finding:** accuracy is flat (29.62-29.73%) across the entire sweep while
perplexity moves 29%. The concept supervision is acting on the model's **calibration**, not on its
top-1 answers. The readable model gets the next word right about as often as the plain one; it is
just much less sure of itself.

## THE STRUCTURAL IDEA THAT FOLLOWS

The concept targets are **computable from the text**. `Concept.labels(X)` is a pure function -- it
does not need a model at all. So the concept layer does not have to be learned through the language
loss. It can be trained **directly on its own targets**, and then frozen.

That removes the conflict by construction: two objectives fighting over the same parameters become
one objective each, on different parameters.

**Two-phase plan, pre-registered before running:**

* **Phase A:** train the embedding + concept layer on the concept targets ONLY. Pure supervised
  classification, no language modelling. Fast.
* **Phase B:** FREEZE the embedding and concept layer. Train ONLY the answer layer on next-token
  prediction. No concept loss at all, because the concepts are already fixed.

**PREDICTION:** the perplexity gap should fall well below +12.281, because the answer layer is
trained with nothing competing with language modelling. The answer layer sees fixed concepts, so
it has less to work with.

**PREDICTION OF THE RISK, stated now:** fixing the concepts means the answer layer cannot adapt
them, so accuracy may DROP below 29.65% even as perplexity improves. If that happens, the result is
"perplexity fixed at the cost of accuracy" and must be reported as such -- not as a win.

**THE BARS DO NOT MOVE.** AUROC 0.55, firing floor 2%, 9/9 concepts, and the gap against the
matched normal (77.804) reported as measured.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
