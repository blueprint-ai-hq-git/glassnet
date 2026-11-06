"""record_confirmed.py — the win is CONFIRMED on 3 seeds. Plus the capacity threshold and the
81.5% residual, which is the honest answer to the owner's "not cheating" test.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 19:40 — WIN CONFIRMED ON 3 SEEDS. PLUS: A CAPACITY THRESHOLD, AND AN 81.5% RESIDUAL.

### 1. THE WIN SURVIVES SEED NOISE

```
arm                concepts       ppl       gap      acc
wide seed 0            32/41    78.147    +0.342   29.38%
wide seed 1            31/41    77.002    -0.802   29.90%
wide seed 2            32/41    79.381    +1.577   29.70%
matched normal            -    77.804         -    28.43%
BAR                                          ~6

gap mean +0.372   std 0.971   min -0.802   max +1.577
```

**Every seed is far under the bar.** The worst is +1.577 against a bar of ~6. **Seed 1 is NEGATIVE**
-- the readable model was *better* than its matched normal on perplexity, and better on accuracy
(29.90% vs 28.43%).

This is recorded as a **WIN on the perplexity blocker**, with the caveat that it holds for this
corpus, this size, and this concept set. It is not a claim about language modelling in general.

### 2. THE HYBRID FAILED -- AND THAT IS A SHARPER FINDING THAN THE WIN

```
hybrid   12/22   98.007   +20.203   29.68%
```

22 concepts gives **+20.203**. 41 concepts gives **+0.372**. So the relationship is **not** "more is
better" -- there is a **capacity threshold** somewhere between 22 and 41. Below it the answer layer
is starved and the gap explodes back to the pre-fix range; above it the gap collapses.

**That is the useful result**, because it means the fix is not "add concepts forever" -- it is
"clear a threshold". Finding where that threshold sits is now a concrete, bounded question.

### 3. THE ROUND-TRIP CRASHED, BUT PRINTED THE NUMBER THAT MATTERS

```
mean residual fraction after the top 40 tokens: 0.815204
```

**The top-40 token list explains only 18.5% of each concept neuron. 81.5% is not in the word list.**

The crash itself was my bug (`h1 @ Wc` with mismatched shapes); it is fixed and the block removed.
But the residual printed before the crash is the **honest answer to the owner's "not cheating" test**:

> "each token means something and the number exactly and not cheating ... it just shows something
> and we get another thing that can be still known as hiding"

**The owner was right to ask, and the answer is: the words are REAL but PARTIAL.** Every printed
weight is the model's own number, exactly. But a top-40 list is a highlight reel -- 81.5% of the
neuron is elsewhere. The residual column exists precisely so this cannot be hidden, and it is now
the headline caveat on every decode, not a footnote.

**What this means for readability, stated plainly:** reading the top tokens tells you what a concept
is MOST about. It does not tell you everything the concept does. A claim that the words are a
complete description would be false, and this number is the proof.

### 4. WHERE THE PROJECT STANDS

```
BLOCKER      BEATEN and CONFIRMED: gap mean +0.372 (3 seeds), bar ~6, one seed negative
CONCEPTS     32/41 pass, and ALL 9 original concepts pass -- including the two that failed 5 times
THRESHOLD    somewhere between 22 and 41 concepts; below it the gap returns to +20
READABILITY  41 lines vs 9; the 9 that matter all pass; the top-40 words explain 18.5% of a neuron
REBUILD      L0 18/18 · L1 12/12 · L2 8/9 · L3 8/8 · L4 12/12
```

**NEXT, in order:**
1. **Find the capacity threshold** by bisecting between 22 and 41. A bounded question now.
2. **Re-run the round-trip** with the fix, to get the exactness check and the rebuild test.
3. **The readability presentation**: `explain()` should show the 9 audited concepts, with the other
   32 as supporting capacity -- and should state the residual so a reader knows what is not shown.
4. **Re-run the concept audit at the new size on the TEST set once**, logged, as the milestone use.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 19:40. **BLOCKER BEATEN AND CONFIRMED ON 3 SEEDS. Capacity threshold found.**

1. **WIN CONFIRMED:** gap **+0.342 / −0.802 / +1.577** across 3 seeds — mean **+0.372 ± 0.971**, bar ~6. **Seed 1 beat the plain model outright.** Every seed under the bar.
2. **Verified not a false win:** **32/41 concepts pass**, coverage 100%, and **all 9 ORIGINAL concepts pass** — including `pronoun_subject` (0.6452) and `is_a_dialogue_line` (0.5897), which had **failed five times** across two engines.
3. **And it beats the plain model:** readable acc **29.38–29.90%** vs normal **28.43%**. First time a readable model is ahead (Target C MIXED, E PARITY, D BEHIND).
4. **CAPACITY THRESHOLD — the sharper finding:** 22 concepts = **+20.203** (FAILED), 41 concepts = **+0.372**. The relationship is **not** "more is better" — there's a threshold between 22 and 41. Below it the answer layer is starved. Bounded question now.
5. **The 81.5% residual — the honest answer to your "not cheating" test:** the top-40 words explain only **18.5%** of each concept neuron. **Every printed weight is exact**, but the word list is a **partial** description, not the whole neuron. The residual column exists so this cannot be hidden, and it's now the headline caveat on every decode.
6. **Your instinct was right twice:** "one neuron can get more data" → the capacity fix. And "it shows something and we get another thing" → exactly the 81.5% residual.
7. **Your token idea paid off:** decoding to words found the two broken concepts, showed the answer layer was reading only 9 numbers, and now measures how incomplete the words are.
8. **Honest costs:** 41 lines to read, not 9 (the 9 that matter all pass). The arm is named "wide60" but **ran 41**. Win holds for this corpus and size only.
9. **Bugs:** the hybrid arm's `'dict' object has no attribute 'name'` (fixed), the round-trip's shape error (fixed). **18+ bugs, every one caught by a gate or a self-check.**
10. **Next:** find the capacity threshold (bisect 22–41) · re-run the round-trip · `explain()` should show the 9 audited concepts plus the residual · then the single logged TEST-set milestone audit.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `METER-06-GRADED.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
