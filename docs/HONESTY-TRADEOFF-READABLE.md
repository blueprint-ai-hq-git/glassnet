# Trade-off found: forcing the concepts to carry the answer makes explanations faithful but concept names WORSE

Date: 2026-11-14 20:02. Tool `tools/chat_concept_independent_audit.py`, same 22 concepts, same held-out synonym lists, same 4,000
windows as the published direct-path audit; only the checkpoint changed. Predicted in
`lens/runs/READABLE-CONCEPT-HONESTY-PREDICTIONS.txt` BEFORE the run; output `lens/runs/READABLE-CONCEPT-HONESTY.txt`.

## The comparison

```
checkpoint                         own-label median   held-out synonym median   gap      held-out below 0.60
direct path, d512, 9 ep               0.9097               0.6980             0.2117        6 of 22
READABLE (no-direct), d512, 9 ep      0.7193               0.5392             0.1801       16 of 22
```

Both arms are d512, 9 epochs, same corpus, same seed, same instrument. **The readable model's concepts are markedly worse at both
jobs**: -0.19 against their own trained labels and -0.16 against unseen synonyms. Sixteen of 22 concepts now score below 0.60 on
held-out synonyms, against 6 of 22 for the direct-path model.

## Predictions, all but one failed

- **P1 FAILED**: I predicted the readable checkpoint would score HIGHER than 0.6980 on held-out synonyms. It scores **0.5392**.
- **P2 FAILED**: I predicted its own-label median would stay within 0.05 of 0.9097 (as a sanity condition). It is **0.7193**, i.e.
  0.19 lower.
- **P3 HOLDS, but for the wrong reason**: the gap does shrink (0.2117 -> 0.1801), yet only because **both** numbers fell. A smaller
  gap between a weak own-label score and a weaker held-out score is not better semantics.
- Falsifier ("held-out at or below 0.6980 means forcing the concepts does not make them more name-honest") **FIRED**.

## What this actually means

The project now has two different kinds of honesty, and they pull against each other:

```
                    explanation faithfulness            concept name semantics
direct path         NOT exact (0 of 200 windows,        BETTER  (own 0.9097 / held-out 0.6980)
                    concepts carry 8.9-18%)
readable            EXACT (100 of 100 windows,          WORSE   (own 0.7193 / held-out 0.5392)
                    concepts carry 99.999996%)
```

**Making the concept vector the only route to the answer buys an exact, complete explanation and costs concept quality.** The most
plausible mechanism is that 36 concepts now have to encode everything needed to predict 12,002 tokens, so each neuron is pressed
into service for information unrelated to its declared name - which is exactly what a drop in own-label AUROC measures.

This matters directly for the "100% honest and understandable" goal: **the readable chat model is honest about the arithmetic of its
own decision, and its concept names are less trustworthy than the direct-path model's.** Reporting only the exactness would have
been a half-truth; reporting only the concept scores would have missed that the readable model can account for its answer at all.

## Limits and next questions

- One seed per arm, d512, 9 epochs. The 47M d1024 checkpoints were not audited for this document.
- The trainer's own fidelity audit reports **34 of 36 pass** for this readable checkpoint, at its 0.55 bar - consistent with a median
  of 0.7193 and a spread sitting just above that bar. The independent-label instrument is the stricter one and it disagrees.
- **The honest test of the trade-off**: widen the concept layer. If concept quality recovers when there are 96-128 concepts to share
  the predictive load, the mechanism above is confirmed and both kinds of honesty can coexist. That is a definition change and waits
  on the owner.

## Follow-up the same evening: the 2x2 at matched epochs - the trade-off is NOT a width artefact

The earlier comparison was lopsided (the readable arm had been audited at d512/9ep and d1024/18ep, the direct-path arm at d512/9ep
and d1024/9ep). The missing cell was measured to complete a clean 2x2 at matched epochs. Predictions
`lens/runs/CONCEPT-HONESTY-2X2-PREDICTIONS.txt`, output `lens/runs/CONCEPT-HONESTY-2X2.txt`.

```
                    own-label median            held-out synonym median
                 d512      d1024   change      d512      d1024   change
direct path     0.9097    0.9754   +0.0657    0.6980    0.7178   +0.0198
readable        0.7193    0.7256   +0.0063    0.5392    0.5436   +0.0044
```

**Three conclusions, all from one table:**

1. **Width helps the direct-path concepts and does almost nothing for the readable ones** (+0.0657 vs +0.0063 on own labels; +0.0198
   vs +0.0044 on unseen synonyms). This is the same shape as every other width result tonight: width buys the direct-path model
   things it cannot buy the readable model, because the readable model's constraint sits *upstream* of the width.
2. **The readability cost to concept names is large at both widths**: -0.18/-0.25 on own labels and -0.16/-0.17 on held-out synonyms.
   So it is **not** a d512 artefact and cannot be dismissed as a small-model effect.
3. **More training recovers part of it** (0.7193 -> 0.8384 own-label from 9 to 18 epochs at d1024) but not the meaning gap, which
   stays near 0.57 against the direct-path model's 0.72.

Predictions scored: **P1 HOLDS** (d1024 readable own-label 0.7256, inside the predicted 0.72-0.84 band, at its low end), **P2 HOLDS**
(held-out 0.5436, inside 0.54-0.70), **P3 HOLDS** (at matched width and epochs the readable arm is worse on both metrics). The
falsifier - readable meeting or beating the direct-path arm, which would have made this a d512 artefact - **did not trigger**.

## The single sentence this all reduces to

**The 36-concept bottleneck caps the readable chat model on all three dimensions at once - accuracy, vocabulary breadth, and the
meaning of its own concept names - and none of those three caps can be relieved with more width, because the constraint sits upstream
of the width.** The only untested lever left is the one that widens the constraint itself.
