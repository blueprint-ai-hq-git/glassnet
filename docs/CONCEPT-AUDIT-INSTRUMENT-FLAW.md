# The concept fidelity audit passes the wrong concepts - and fails the ones it should reward

Date: 2026-11-14, same night as the lam-concept sweep. Source logs `logs/exposure/lamconc_0p{1,3,6}.log`, extracted table
`lens/runs/CHAT-LAMCONCEPT-PERCONCEPT.txt`.

## The bar is much lower than the headline implies

`lens/api.py:176`:

```python
def audit(net, X, concepts=None, auroc_bar=0.55, fire_floor=0.02):
    ...
    ok = bool(au >= auroc_bar and fires >= fire_floor)
```

So a concept "PASSES" if its AUROC clears **0.55** - barely above chance - and it fires on at least 2% of windows. The published
headline from the sweep ("31 of 36 concepts mean their name") is therefore **31 of 36 are better than barely-better-than-chance**,
which is a much weaker statement than it reads.

## The honesty ladder at every bar

```
lam        >=0.55*   >=0.60   >=0.70   >=0.80   >=0.85
0.1           19       13        4        2        0
0.3           25       19        8        4        3
0.6           31       25       13        4        3
* trainer's own PASS rule (auroc >= 0.55 AND fires >= 0.02)
```

At a bar that would mean something (0.70), **lam 0.6 is 13 of 36 (36%)**, and only **3 of 36 (8%)** reach 0.85. The direction of the
sweep result is unchanged - pressure still moves every column up - but the absolute honesty claim must be quoted from this ladder,
never from the single 31/36 figure.

## The instrument fails accurate rare concepts and passes degenerate ones

Exactly one concept fails at a **high** AUROC:

```
mentions_a_number   lam 0.1  AUROC 0.77480  fires 0.00850   fires below floor
                    lam 0.3  AUROC 0.83267  fires 0.01350   fires below floor
                    lam 0.6  AUROC 0.84468  fires 0.01700   fires below floor
```

It is **well detected** at 0.845 and is failed only because it fires on 1.7% of windows instead of 2%. Meanwhile seven concepts pass
**only** because the AUROC bar is 0.55, and every one of them fires on 90-97% of windows:

```
mentions_time        AUROC 0.55603  fires 0.91625
mentions_family      AUROC 0.56518  fires 0.90725
mentions_a_quantity  AUROC 0.57052  fires 0.95750
mentions_science     AUROC 0.57776  fires 0.69975
mentions_work        AUROC 0.57850  fires 0.95475
mentions_place       AUROC 0.58323  fires 0.94725
mentions_sport       AUROC 0.59890  fires 0.96700
```

A detector that fires on 95% of everything is nearly constant, and a nearly constant detector can score a slightly-above-chance AUROC
for free. **So the audit currently rewards the degenerate concept and punishes the rare accurate one** - the exact inversion of what a
"does this name mean what it says" check is for.

## What this does not change

- The **direction** of the lam result stands: every column of the ladder rises with pressure (19→25→31 at 0.55, 13→19→25 at 0.60,
  4→8→13 at 0.70, 0→3→3 at 0.85).
- `mentions_law` at **AUROC 0.474 (below chance)** with support 0.0090 corroborates the earlier finding that its support was mostly
  the ambiguous word "right": with that token removed, the model cannot rank the concept at all. It should be treated as a concept
  the corpus barely contains, not as a training failure.

## Next work, in order

1. **Report the ladder, not the single number**, everywhere concept honesty is quoted.
2. **Fix the instrument**: replace "fires >= 0.02" with a rule that does not punish rare concepts (e.g. bar AUROC only, and report the
   positive rate beside it), and add a **constant-detector guard** so a concept firing on >0.90 of windows cannot pass on a 0.56 AUROC.
3. Re-run the audit on the existing checkpoints with the fixed instrument to see how the honest count moves on both sides.

## Correction to what the sweep claims, after rebuilding the verdict

I patched `lens/api.py` so every concept row also carries a **strict** verdict (the old `pass` is untouched, so no published number
moves). Strict = AUROC bar only, no rarity penalty, and any concept firing on more than 90% of windows must clear **0.75** instead of
0.55. Self-test `tools/test_strict_audit.py`, 7 of 7 cases pass.

```
lam        old pass   strict pass   median AUROC   mean AUROC
0.1          19/36      20/36          0.56490       0.58931
0.3          25/36      18/36          0.61657       0.62950
0.6          31/36      21/36          0.67186       0.66788
```

At lam 0.6 the strict instrument **demotes 11 concepts** and promotes 1:

```
DEMOTED - near-constant and weak (old pass, strict fail)
  mentions_time 0.55603 f0.916   mentions_family 0.56518 f0.907   mentions_a_quantity 0.57052 f0.958
  mentions_work 0.57850 f0.955   mentions_place  0.58323 f0.947   mentions_sport      0.59890 f0.967
  mentions_body 0.63152 f0.923   mentions_animals 0.64550 f0.907  mentions_travel     0.67738 f0.916
  is_imperative 0.73098 f0.943   is_a_denial     0.74966 f0.912
PROMOTED - accurate but rare (old fail, strict pass)
  mentions_a_number 0.84468 f0.017
```

So two corrections to my own sweep claim:

1. **The threshold-free effect is real and monotone**: median AUROC rises **0.56490 -> 0.61657 -> 0.67186** with lam 0.1 -> 0.3 -> 0.6.
   Concept pressure genuinely improves how well the layer ranks each concept. This survives the rebuilt verdict.
2. **The "19 -> 25 -> 31 of 36" table is not a usable honesty metric.** Under a corrected instrument the count is **20 -> 18 -> 21**,
   i.e. no monotone improvement, because most of the growth at the 0.55 bar was near-constant detectors becoming slightly less
   arbitrary. The claim "the default leaves 6 concepts on honesty for nothing" does **not** survive - what survives is "pressure raises
   AUROC, and the binary pass count was measuring the wrong thing".

Also newly visible and worth its own line: **11 of 36 chat concepts (31%) fire on more than 90% of windows at lam 0.6.** That is a
structural property of the concept layer, not a training-pressure artifact, and it is a better explanation of thin explanations than
the five "failures" that started this thread.
