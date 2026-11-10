# Capacity verdict: doubling the width bought 0.50 pp, and it all went to the unreadable path (2026-11-10)

One change against the 25.52% baseline: d=512 -> d=768, 15.57M -> 29.42M parameters, same matched split, same recipe,
3 epochs. Prediction on record: 1-3 pp if capacity binds.

```
epoch        d512 baseline        d768           change
1              19.69%            20.30%         +0.61 pp
2              24.26%            24.16%         -0.10 pp
3              25.52%            26.02%         +0.50 pp
final VAL ppl   99.92             96.20         better
concepts         --              35/36          coverage 100%
epoch time      205 s             246 s         1.2x
```

**Prediction not met.** +0.50 pp for 89% more parameters, against a project-measured single-seed spread of 0.26-3.07 pp.
The trainer's own audit had already flagged why: at 2,000,000 windows this arm is **UNDER-TRAINED, 2.18 tokens per
parameter**, against 4.11 for the d512 arm. So the honest reading is not "capacity is irrelevant" but "capacity and
data budget are tangled, and this arm changed one without the other".

## The part that matters more: where the extra capacity went
Same decomposition harness as the d512 measurement (`tools/concept_only_eval.py`, 600 matched validation windows):

```
                  d512      d768
full             22.17%    23.17%
copy path only   20.17%    21.50%
concept only      3.00%     4.67%
concept agrees with the full model's answer   8.2%  ->  4.3%
price of making the concepts decide         19.17 pp -> 18.50 pp
```

Every path improved a little, and the one that improved MOST relative to its starting point is the copy path. The
concept path does rise from 3.00% to 4.67% - it is no longer below the 4.11% "always answer the most frequent token"
baseline - but its agreement with the model's actual answers FALLS from 8.2% to 4.3%.

**So widening the model makes it less readable, not more.** Two and a half times the readable path's accuracy was
bought, and the model's decisions moved further away from what its 36 named concepts would say. That is the opposite
of the direction this project is supposed to be going, and it happened by default rather than by choice.

## What this settles
1. **Do not add width for accuracy.** At this budget the return is inside noise and the accuracy that does arrive is
   copy-path accuracy.
2. **The readable path needs its own objective, not more parameters.** It received 89% more capacity and still decides
   4.3% of answers. Anything that is supposed to make this model readable has to be trained for directly.
3. **The next arm should give the concepts the decision on purpose** - train with the copy path off - and report the
   accuracy that survives. That number is the real price of readability and nobody has measured it yet: 18.50 pp is
   the price of retrofitting the concepts onto a copy-trained model, which is not the same thing as training a model
   whose concepts are its only route to the answer.
