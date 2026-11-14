# At 9 epochs the readable chat model matches its control on accuracy - the remaining deficit is vocabulary, not accuracy

Date: 2026-11-14. Readable arm `nodirect9_d512_s0` and matched control `direct9_d512_s0`: identical except `--direct
--direct-mode tied`, both 9 epochs, d512, seed 0, 600k windows of the full 294,225,090-token corpus, each trained fresh inside one
script so the comparison is matched in this run. Predictions `lens/runs/NODIRECT-9EP-PREDICTIONS.txt` written BEFORE launch.
Logs `logs/exposure/{nodirect9,direct9}_d512_s0.log`, outputs `lens/runs/NODIRECT-9EP.txt`, `NODIRECT9-VS-CONTROL.json`,
`DIVERSITY-nodirect9_d512_s0.json`.

## The matched pair

```
                                   readable   control    gap
final VAL accuracy                   19.95%    20.12%   -0.17
final VAL ppl                        292.65    352.11       -
held-out best (by position)          21.45%    23.67%   -2.22
next-token metric, 2000 positions    21.90%    22.40%   -0.50
distinct tokens predicted               44       191      -147
<unk> share                           0.0%      0.0%        -
concept fidelity audit               34 of 36  36 of 36     -2
```

**The accuracy cost of readability has almost vanished: -0.17 points of validation accuracy at the final epoch, and -0.50 on the
next-token metric.** At 3 epochs the same comparison was -4.67 points VAL (`docs/history/READABLE-CHAT-NODIRECT-ARM.md`), so
**4.5 of those 4.67 points were under-training, not readability.**

The readable arm also generalises better in one sense: its validation perplexity at the end is **292.65 against the control's
352.11**. The control's validation perplexity *rises* over the last epochs (252 -> 352, from
`logs/exposure/direct9_d512_s0.log`) while its training accuracy climbs 36.9% -> 44.3%, which is textbook overfitting. The readable
arm's perplexity stays low because the 36-concept bottleneck is a regulariser.

## Predictions, scored

- **P1 HOLDS**: I predicted the readable arm would close to within 3.0 points of its control at 9 epochs. It closed to **-0.17**
  (final VAL) and **-2.22** (held-out best).
- **P2 FAILED**: I predicted more than 60 distinct tokens. It reached **44** (from 28 at 3 epochs), still short of my bar.
- **P3 HOLDS**: `explain()` stays exact on the 9-epoch readable checkpoint - mean residual **5.634e-07**, max **2.295e-06**, exact
  on **100 of 100** windows, `concept_share` **99.999996%**. Readability is not a 3-epoch artefact.
- Decision rule from the predictions: accuracy is **99.2% of the control** (well past the 80% bar) but diversity is **44, below the
  60 bar** - so the rule points at capacity for diversity, and at nothing else.

## Where the deficit now sits

```
what                 readable (9 ep)   control (9 ep)   3-epoch pair
accuracy gap             -0.17 VAL        --            -4.67 VAL
diversity                 44 tokens      191 tokens     28 vs 121
concept fidelity          34 of 36       36 of 36       23 vs 28
explanations              EXACT 100/100  not exact      exact
```

**Accuracy is solved for this design at this size. The one remaining deficit is vocabulary breadth: the readable model speaks in 44
distinct words where its control uses 191.** That is consistent with, and stronger evidence for, the capacity diagnosis: 15 of 36
concepts fire on a median window at 9 epochs (`DIVERSITY-nodirect9_d512_s0.json`), and 36 half-saturated numbers cannot address
12,002 output tokens.

## What this changes

The readable chat model is now a **credible** model, not a demonstration: within 0.17 points of its own control on validation
accuracy, better validation perplexity, exact explanations on every window tested, 34 of 36 concepts honouring their names. What it
cannot do yet is talk with a wide vocabulary.

**Next, in order:**
1. **Widen the concept layer** (96-128 concepts). This is the measured bottleneck and the only remaining accuracy-side deficit. It
   changes the concept definitions, so it waits on the owner.
2. Re-run this matched pair at d1024 to check the width interaction.
3. Any full-length claim about the readable chat model must quote **accuracy and diversity and concept fidelity together**, because
   accuracy alone now looks finished while the vocabulary deficit is invisible in it.
