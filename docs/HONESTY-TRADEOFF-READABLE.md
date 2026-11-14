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
