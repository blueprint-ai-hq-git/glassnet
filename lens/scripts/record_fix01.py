"""record_fix01.py — FIX-01: both predictions correct, and my own diagnosis was wrong.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 15:10 — FIX-01: 9/9 CONCEPTS. BOTH PREDICTIONS CORRECT. MY DIAGNOSIS WAS WRONG.

Pre-registered in `FIX-01.md` before the run. One change: `lam_concept` 0.1 -> 1.0.

```
concept                  lc=0.1            lc=1.0
pronoun_subject        0.5443 F 0.273  ->  1.0000 P 0.328
is_a_dialogue_line     0.5066 F 0.331  ->  1.0000 P 0.181
pronoun_object         1.0000 P 0.147  ->  1.0000 P 0.147
is_a_negation          0.6527 P 0.164  ->  1.0000 P 0.093
is_archaic_address     1.0000 P 0.084  ->  1.0000 P 0.083
contains_a_wh_word     1.0000 P 0.047  ->  1.0000 P 0.047
mentions_speech        1.0000 P 0.045  ->  1.0000 P 0.045
mentions_giving        1.0000 P 0.023  ->  1.0000 P 0.023
mentions_knowing       1.0000 P 0.045  ->  1.0000 P 0.045

pass rate                 7/9     ->    9/9
val ppl                 90.085    ->  115.829     (gap +12.281 -> +38.025)
val acc                 29.73%    ->   29.65%
```

**PREDICTION 1 -- CORRECT.** Both concepts reach AUROC >= 0.55. Both hit **1.0000**. So do the
other seven. **9/9 concepts.**

**PREDICTION 2 -- CORRECT.** The readable model's perplexity gap gets worse: **+38.025**, against
+12.281 at the weaker setting. Three times worse, as predicted.

**PREDICTION 3 -- DOES NOT APPLY, and this is where I was wrong.** Prediction 3 said that if the
two still failed, the cause would be the concept DESIGN. They did not fail -- so the design was
never the problem.

**MY DIAGNOSIS FROM LAST NIGHT IS WITHDRAWN.** I wrote that the two concepts were "competing for
the same signal" and that the layer had "no room for two overlapping ones". That was wrong. The
evidence now says plainly: **the concepts were UNDER-TRAINED.** `lam_concept=0.1` made the concept
supervision one tenth as strong as the language-modelling loss, and two concepts with the highest
support (25.07%, 23.19%) were the ones that lost that fight -- not because they conflicted with
each other, but because they had the most work to do and the least pressure to do it.

The decoding-into-words work showed **what** was wrong (character names instead of pronouns;
signs flipped on the speaker labels). It did **not** show **why**, and I over-read it. The
distinction matters: the words were right, my explanation of them was wrong.

**THE EXCHANGE RATE, NOW MEASURED**

```
7/9 concepts  ->  gap +12.281
9/9 concepts  ->  gap +38.025
```

**Two more honest concepts cost +25.744 perplexity.** That is the price of readability, stated as a
number instead of a slogan. The old project's Target C -- a much weaker concept layer -- had
`pronoun_subject` at 0.9982 and a MIXED perplexity result; this is the same trade-off seen from the
other end.

**Note also:** accuracy barely moves (29.73% -> 29.65%) while perplexity changes by 29%. The
readable model gets the next word right about as often but is much less well calibrated. Whatever
the concept supervision is doing, it is doing it to the CONFIDENCE, not to the top-1 answer.

**NEXT: the sweep.** 0.1 gives 7/9 at +12.281; 1.0 gives 9/9 at +38.025. `sweep_lc.sh` is running
0.3 and 0.5 to find the cheapest setting that still reaches 9/9. Nothing is overwritten: each arm
writes to `lens/runs/fix01_lc<value>/`.

**Pre-registration for the sweep (written before it finished):** I expect 9/9 to be reached
somewhere between 0.3 and 0.5, and the gap to scale roughly with `lam_concept`. If a setting gives
9/9 at a gap under +10, that is a genuine win on the blocker -- the first one.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "TARGET-LEDGER.md")
txt = open(p).read()
add = """## FIX-01 — 9/9 CONCEPTS ACHIEVED. THE PRICE IS +25.7 PERPLEXITY.

One change, pre-registered: `lam_concept` 0.1 -> 1.0.

```
                      lc=0.1        lc=1.0
concepts passing        7/9    ->     9/9
pronoun_subject      0.5443 F   ->  1.0000 P
is_a_dialogue_line   0.5066 F   ->  1.0000 P
val ppl              90.085     -> 115.829
gap                  +12.281    ->  +38.025
```

**The two concepts that failed on the old engine and the new one, in Targets D, E, G2, I and the
rebuild, are NOT broken. They were under-trained.** At `lam_concept=1.0` both score a perfect
1.0000, and so do the other seven.

**The diagnosis recorded on 2026-10-07 02:00 ("they compete for the same signal; the layer has no
room") is WITHDRAWN.** The cause was supervision strength, not concept design. Decoding the model
into words showed *what* was wrong; the *why* I gave was wrong.

**The exchange rate, measured:** two more honest concepts cost **+25.744 perplexity**. That is the
price of readability as a number. `sweep_lc.sh` is finding the cheapest setting that still reaches
9/9.

"""
anchor = "## WHAT \"MOVING DOWN TO THE NEXT ONE\" MEANS"
assert anchor in txt
txt = txt.replace(anchor, add + anchor, 1)
open(p, "w").write(txt)
print("TARGET-LEDGER.md updated: FIX-01 + the diagnosis withdrawal")
