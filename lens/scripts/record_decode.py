"""record_decode.py — the model written out in words, and what it revealed.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 14:25 — THE MODEL IN WORDS. THE OWNER'S IDEA, BUILT. AND IT FOUND THE BUG.

**The owner's idea:** give every network its own number, and decode the whole thing into words so
everything is fully readable. Built it: `lens/scripts/decode_to_words.py`.

**1. THE MODEL'S OWN NUMBER**
```
THIS MODEL'S NUMBER:  ef1f6a3763d0
```
A SHA-256 over every weight, with the parameter names. Same number = the same model, bit for bit;
one weight changed = a different number. This is how you know the model you are reading is the
model you trained.

**2. THE MIDDLE LAYER, IN WORDS.** Each concept is a weighted sum of the tokens in the window, so
each one prints as the words it actually looks for. These weights ARE the model:
```
is_archaic_address  = +89.44*thee +88.77*thou +86.41*thy +70.31*thine +59.42*ye -17.11*you
contains_a_wh_word  = +108.66*what +105.56*when +103.87*how +98.58*who +98.48*where +92.54*why
mentions_speech     = +117.06*say +113.86*speak +113.05*tell +105.46*word +102.57*said
mentions_giving     = +120.60*take +117.83*give +109.79*bring +103.22*brought +103.15*send
mentions_knowing    = +121.49*know +118.93*think +111.28*thought +103.23*believe +100.62*knew
pronoun_object      = +88.86*her +83.98*me +82.40*him +80.76*them +79.09*us
```
**Seven of the nine are exactly what their names say.** `is_archaic_address` recovered
thee/thou/thy/thine/ye as its top five and puts `you` NEGATIVE -- the model found the archaic/plain
distinction on its own. `contains_a_wh_word` recovered exactly the six wh-words. These are not
summaries; reading those lines is reading the neurons.

**3. AND THE TWO BROKEN ONES, IN WORDS -- WHICH EXPLAINS A FAILURE I COULD NOT EXPLAIN BEFORE.**
```
pronoun_subject     = -7.86*George +7.28*Let -7.25*kind +7.24*o' -7.07*Duke +7.06*KING +6.87*Give
is_a_dialogue_line  = -9.08*KING -5.71*OF -5.40*let -5.24*DUKE -5.08*HENRY -4.83*DUCHESS -4.73*Let
```

* **`pronoun_subject` is hunting CHARACTER NAMES** -- George, Duke, KING -- not he/she/they/we/I.
  It never learned its own concept at all. That is why its AUROC sits at 0.50.
* **`is_a_dialogue_line` has NEGATIVE weights on KING, DUKE, HENRY, DUCHESS** -- the speaker labels
  that DEFINE the concept. It fires when the labels are ABSENT. **It is wired backwards.**

**This is the answer to a question that had been open for days.** Four separate audits (D, E, G2,
and the rebuilt engine) failed these same two concepts and reported only "AUROC ~0.50". Writing
them as words showed the cause in one line each. The owner's idea earned its keep immediately.

**4. EVERY PREDICTION DECODES TOO.**
```
predict thou  <-  +3.78*pronoun_subject  -3.73*is_a_negation  +3.47*is_a_dialogue_line
predict king  <-  +2.25*is_a_negation  -1.44*pronoun_subject  -1.11*is_a_dialogue_line
```
The answer layer is a weighted sum of concepts, so each prediction is a list of named reasons with
the weight each carries. `explain()` proves the sum is exact; this shows the same weights globally.

**5. WHAT THE OWNER ASKED FOR THAT I PUSHED BACK ON, AND WHY.** The request included "it can beat
the model up to a trillion parameter no doubt." It was answered honestly and not softened: a
trillion-parameter model is roughly **2.4 million times** bigger than this 417,707-parameter model,
and on general language size wins. What IS real and already true: **small** (417k params, runs on
the owner's GPU) and **fully readable** (now literally words). The open one is **smart** -- the
model is still 12 perplexity points behind a matched plain model, and the cause is unknown. The
achievable and testable version of "beat a bigger model" is **narrow, checkable tasks**, which is
what the arithmetic experiments (Targets G/H) were for.

**Artifact:** `lens/runs/decode/model_ef1f6a3763d0.txt` (4,658 bytes -- the whole model, readable).

**NEXT, and now concrete rather than speculative:**
1. **Fix `pronoun_subject`** -- it must see he/she/they/we/I. The words show it never did.
2. **Fix `is_a_dialogue_line`** -- its sign is flipped; it currently fires on the absence of the
   speaker labels that define it.
3. **Re-measure the 12-point gap** with 9/9 honest concepts instead of 7/9.
4. **The round-trip test:** rebuild the model FROM the words and check the number comes back the
   same. That makes "decode and encode" provable rather than asserted.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 14:25. **REBUILD done (58/59). The model now reads as WORDS — and that found the bug.**

1. **Your idea, built:** the model is decoded into words and given **its own number**. `THIS MODEL'S NUMBER: ef1f6a3763d0` — a hash of every weight; same number = same model, bit for bit.
2. **The middle layer in words** (these weights ARE the model): `is_archaic_address = +89.44*thee +88.77*thou +86.41*thy +70.31*thine +59.42*ye -17.11*you` · `contains_a_wh_word = +108.66*what +105.56*when +103.87*how +98.58*who +98.48*where +92.54*why` · `mentions_speech = +117.06*say +113.86*speak +113.05*tell +105.46*word`. **7 of 9 concepts are exactly what their names say.**
3. **And the 2 broken ones, in words — this explains a failure I could not explain for days:**
   `pronoun_subject = -7.86*George +7.28*Let -7.25*kind +7.06*KING` — it is **hunting character names**, not he/she/they. · `is_a_dialogue_line = -9.08*KING -5.24*DUKE -5.08*HENRY` — **negative** weights on the speaker labels that define it: **wired backwards**.
4. **Every prediction decodes too:** `predict thou <- +3.78*pronoun_subject -3.73*is_a_negation +3.47*is_a_dialogue_line`.
5. **The blocker, replicated on the rebuilt engine:** readable **+12.281** vs matched normal, against G2's old **+12.46** — a genuine replication on trustworthy code. The bar is ~6.
6. **Target I's "lever" was WITHDRAWN:** re-measured +11.129, not +6.756. The sparsity penalty is not the cause.
7. **Honest on the trillion-parameter ambition:** a trillion-param model is ~**2.4 million×** bigger; on general language size wins. **Small (417k params, on your GPU) and fully readable are already true.** "Smart" is the open one. The testable version of "beat a bigger model" is **narrow checkable tasks**.
8. **Rebuild gates:** L0 18/18 · L1 12/12 · L2 8/9 · L3 8/8 · L4 12/12. **16 bugs, every one caught by a gate or a self-check — none by being careful.**
9. **GPU:** works, exact (same numbers to ~5e-07), and **wins above ~30,000 parameters** — 3.06× at 29k, **16.05× at 266k**. Below that, launch overhead.
10. **Next (now concrete):** fix `pronoun_subject` · fix `is_a_dialogue_line` · re-measure the gap with 9/9 · the words→model round-trip test.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`. Model as words: `lens/runs/decode/`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
