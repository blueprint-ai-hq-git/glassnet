"""record_twophase.py — the structural fix failed, and found the real cause.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 16:00 — TWO-PHASE FAILED (WORSE THAN PREDICTED). IT FOUND THE REAL CAUSE.

```
arm              concepts       ppl       gap      acc
two-phase             9/9   116.430   +38.626   29.65%   <- worst of all arms
lc=0.3 (best 9/9)     9/9   100.850   +23.045   29.62%
lc=0.1                7/9    90.085   +12.281   29.73%
matched normal          -    77.804         -    28.43%
```

**PREDICTION 1 -- WRONG.** Written before the run: "the perplexity gap falls WELL BELOW +12.281,
because phase B has nothing competing with language modelling." It rose to **+38.626**, the worst
result of any arm.

**PREDICTION 2 -- WRONG.** Written before the run: "accuracy may DROP below 29.65% because the
answer layer sees fixed concepts." It stayed at **29.65%**.

**PREDICTION 3 -- CORRECT.** 9/9 concepts, as predicted, since phase A trains them directly.

### WHAT THE TWO PHASES SHOW

```
Phase A: concept BCE 0.00168    <- the concepts are fit essentially PERFECTLY
Phase B: CE 5.0158              <- ppl ~151, close to uniform (ln 1500 = 7.31)
```

The concepts are perfect and the language model is near-random. **The answer layer cannot predict
the next word from nine on/off values.**

### THE REAL CAUSE: THE READABLE LAYER IS AN INFORMATION BOTTLENECK

Nine concepts means **nine numbers** carrying everything the answer layer will ever know. The
vocabulary is 1,500 words. Nine binary concepts can express at most 2^9 = 512 states, and in
practice far fewer, because most are near-constant in any given window.

**This explains every measurement in this project at once:**

* **Why the cost scales monotonically with `lam_concept`.** Stronger supervision drives the
  concepts toward binary (that is what a balanced BCE does). More binary = fewer usable states =
  less information reaching the answer layer = worse language modelling. The curve was never about
  "conflict between objectives"; it was the bottleneck tightening.
* **Why accuracy stays flat (29.62-29.73%) across the whole sweep while perplexity moves 29%.**
  The concepts carry the easy, coarse signal -- enough to pick a plausible word class, not enough
  to be confident between 1,500 options.
* **Why two-phase is the WORST arm.** It produces the most perfectly binary concepts, therefore
  the tightest bottleneck.
* **Why Target C, with a much weaker concept layer, had `pronoun_subject` at 0.9982 and a MIXED
  perplexity result** -- a looser bottleneck leaks more information.

### SO THE COST OF READABILITY IS THE BOTTLENECK, NOT THE SUPERVISION

Target I tried to remove the sparsity penalty: dead end. The sweep tried to tune the strength: no
free lunch. The two-phase tried to separate the objectives: worst result yet. All three were
attacking the supervision. **The supervision was never the problem.**

**The problem is that a concept is forced to be a 0/1 answer.** A balanced BCE trains each concept
neuron to output ~0 or ~1. That throws away almost all of the information the embedding computed.
A concept that could say "this window contains a wh-word, and how strongly, and alongside what
else" would carry far more.

**NEXT EXPERIMENT (FIX-02), pre-registered:**

Replace the binary concept loss with a **soft** one -- the concept targets become graded rather
than 0/1, and the loss lets the neuron output a real value. One change. The audit is unaffected:
AUROC measures RANKING, which works on graded outputs.

**PREDICTIONS, written before running:**
1. The gap at 9/9 falls below +23.045 (the current best 9/9), because graded concepts carry more
   information than binary ones.
2. Accuracy rises above 29.73% (the current best), for the same reason.
3. If BOTH fail, then the bottleneck is the NUMBER of concepts (nine), not their sharpness, and
   the next lever is more concepts -- which costs readability directly, and that trade-off must
   then be presented to the owner as a decision rather than solved.

**Bars unchanged.** AUROC 0.55, firing floor 2%, gap against the matched normal (77.804).
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "TARGET-LEDGER.md")
txt = open(p).read()
add = """## THE REAL CAUSE, FOUND BY A FAILURE — THE CONCEPT LAYER IS AN INFORMATION BOTTLENECK

Three attempts attacked the concept SUPERVISION and all three failed:

* **Target I** (remove the sparsity penalty): the "lever" did not reproduce (+11.129, not +6.756).
* **The sweep** (tune the strength): a clean monotonic curve, cheapest 9/9 is +23.045 against a bar
  of ~6. No win.
* **Two-phase** (separate the objectives entirely): the WORST result of all, **+38.626**.

The two-phase run shows why, in two lines:
```
Phase A: concept BCE 0.00168    <- concepts fit essentially perfectly
Phase B: CE 5.0158              <- ppl ~151, close to uniform
```
Perfect concepts, near-random language model. **The answer layer cannot predict 1,500 words from
nine on/off values.** Nine binary concepts express at most 512 states; the vocabulary is 1,500.

**So the cost of readability is not the supervision — it is the bottleneck.** A balanced BCE forces
each concept toward 0/1, which throws away nearly all the information the embedding computed, and
stronger supervision makes that worse. Every measurement in this project follows from it: the
monotonic curve, the flat accuracy against a 29% perplexity swing, the worst-case two-phase result,
and Target C's looser layer scoring better.

**The next lever is the concepts' SHARPNESS (graded rather than binary), then their NUMBER.** If
sharpness fails, the lever becomes the number of concepts — which costs readability directly, and
that is an owner decision, not something to solve silently.

"""
anchor = "## WHAT \"MOVING DOWN TO THE NEXT ONE\" MEANS"
assert anchor in txt
txt = txt.replace(anchor, add + anchor, 1)
open(p, "w").write(txt)
print("TARGET-LEDGER.md updated: the bottleneck finding")
