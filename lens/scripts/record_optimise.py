"""record_optimise.py — the optimisation diagnostic: the concept layer is the bottleneck.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 17:20 — OPTIMISATION: THE DIAGNOSTIC FOUND THE BOTTLENECK, AND IT IS WORSE THAN ASSUMED.

The owner said "optimise it, optimisation is key". Optimising without measuring is guessing, so this
measured first. The measurement found a specific, quantified defect.

### WHERE THE PARAMETERS GO (chat run 1: d=128, window 8, K=9)

```
  embedding   V*D   = 12,000 * 128  =  1,536,000   92.9%
  concept     D*W*K = 1,024 * 9     =      9,216    0.6%
  answer      K*V   = 9 * 12,000    =    108,000    6.5%
  TOTAL                             =  1,653,216
```

**93% of the model is the embedding table. The concept layer is 0.6% of it and it is the only path
from input to output.**

### THE BOTTLENECK, QUANTIFIED

```
  the concept layer takes 1,024 numbers in and puts 9 numbers out   ->  114:1
  the answer layer must choose 1 of 12,000 words using only 9 numbers  ->  13.6 bits
```

**And the comparison that matters:**

```
  at the measured WIN:  d=256, K=41  ->  2,048 in / 41 out  =  50:1
  THIS MODEL:           d=128, K= 9  ->  1,024 in /  9 out  = 114:1
```

**The chat model is compressing more than twice as hard as every configuration that FAILED the
scaling test.** The winning arm needed 41 concepts for a *4x larger* input. This model runs 9.

**And the run already queued was worse:** d=384 with K=9 is **341:1** -- nearly 7x more compressed
than anything that has ever failed. It was queued before measuring. That is a real process error:
the run was started on a hunch while the measurement was still pending.

### OTHER FINDINGS

```
  <unk> rate:            1.919% of all tokens   (thrown away at the input)
  words for 90% of tokens:   1,555
  words for 95% of tokens:   3,435
  words for 99% of tokens:   8,537   (vocab is 12,000)
```

**A 3,500-word vocabulary would cover 95% of tokens.** The embedding is 93% of the parameters, so
shrinking the vocabulary frees capacity directly. Ranked as a MEDIUM lever and NOT bundled with the
concept change, so the two can be told apart.

### THE FIX, AS A 2x2 GRID (one change at a time)

```
  run 1 (DONE)   d=128  K= 9   114:1   val ppl 353.28   acc 11.08%   concepts 5/8
  run 2          d=384  K=48    64:1
  run 3          d=384  K= 9   341:1   <- isolates d at the bad ratio
  run 4          d=128  K=48    21:1   <- isolates K at a ratio BETTER than the win
```

The concept set was widened from 12 defined (9 kept) to **46 defined**, all by exact token
membership, all pre-registered in `chat_concepts_wide.py` before any run. Ordinary conversational
categories -- dialogue function, participants, topics, register. **No concept was added because it
looked like it would help a metric.**

### PRE-REGISTERED PREDICTIONS

1. **Run 4 (d=128, K=48) gives the best perplexity** of the four, because 21:1 is a better ratio
   than the 50:1 that won the scaling test.
2. **Run 3 (d=384, K=9) is the worst** at 341:1. **If run 3 beats run 4, the bottleneck theory is
   WRONG and that gets said plainly.**
3. Concept pass rate rises with K; the extra concepts pass at lower fidelity, being rarer topics.
4. **Replies stay incoherent in all four.** Widening the layer should improve perplexity, not
   produce a coherent chatbot -- that needs more training, not a wider layer.

### LEARNING-RATE DECAY ADDED AS ITS OWN SWITCH

`--lr-decay` (default 0 = constant, as in run 1). Perplexity was still falling at the last epoch of
run 1 (523 -> 400 -> 353), so a schedule is the second-ranked lever. It is **off** for the grid runs
so the grid measures the concept change alone.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 17:20. **Optimisation diagnostic: the concept layer is the bottleneck, quantified.**

1. **THE DEFECT, MEASURED:** the concept layer takes **1,024 numbers in and puts 9 out (114:1)**. The scaling test's **winning** arm was **50:1** (41 concepts for 2,048 in). **This model compresses more than twice as hard as every configuration that FAILED.**
2. **And the run I'd queued was worse:** d=384 with K=9 is **341:1** — queued before measuring. A real process error: started on a hunch while the measurement was pending.
3. **Where the parameters go:** embedding **92.9%**, concept layer **0.6%**, answer **6.5%**. The concept layer is 0.6% of the model and the only path from input to output.
4. **<unk> rate 1.919%**; a **3,500-word vocab covers 95%** of tokens (vocab is 12,000) — a direct capacity saving, ranked MEDIUM and kept separate from the concept change.
5. **The fix as a 2×2 grid (one change at a time):** run 1 done (d=128 K=9, ppl 353.28, 5/8) · run 2 (d=384 K=48) · run 3 (d=384 K=9, the bad-ratio control) · run 4 (d=128 K=48, ratio **21:1 — better than the win**).
6. **Concept set widened 12 → 46 defined**, all exact-token, all pre-registered in `chat_concepts_wide.py` before any run. **No concept added because it looked like it would help a metric.**
7. **PREDICTIONS WRITTEN FIRST:** run 4 best, run 3 worst. **If run 3 beats run 4, the bottleneck theory is wrong and I'll say so.** Replies stay incoherent in all four — widening a layer is not the same as training a chatbot.
8. **Learning-rate decay added as its own switch** (off for the grid, so the grid isolates the concept change).
9. **Distillation: works on 1 seed** (student 216k beat its control *and* its 820k teacher); 2 more seeds running.
10. **SCALING — G2 FAILS:** the +0.342 win was size-specific; **does not carry to bigger models.** But the readable model is **more accurate at every width**, growing with size — which is what pointed at the concept bottleneck in the first place.
11. **THE CEILING:** 10M = 3.5 hours, 50M = 3.7 days, **GPT-3 = 10,513 years**. "Talk" and "code" reachable; "better than any other model" is ~5 orders of magnitude away.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
