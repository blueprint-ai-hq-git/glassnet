

---

# FEWER CONCEPTS CUTS CANCELLATION ~48% AND EVERY CONCEPT PASSES AUDIT (2026-11-20 13:09)

The decorrelation sweep had closed with a pre-registered remedy: "fewer, wider concepts". It turned out no new code was needed -
`train_chat.py:939` selects the catalogue by env (`TC_WIDE=1` -> the 48-concept set that survives support filtering as our 36;
`TC_WIDE=0` -> the original 9). So the remedy was testable the same hour.

## Measured, 100k windows, lam_decorr held at 0.5, identical except the catalogue

```
                              36 concepts (d05)     9 concepts (narrow)     delta
held-out accuracy             7.19%                 7.37%                   +0.18
perplexity                    956.79                1275.33                 +318.54   <- worse
concepts passing audit        25 of 36  (69%)       9 of 9  (100%)          +31 points
POSITIVE SUPPLY               38.051x the margin    19.839x the margin      -47.9%
concepts supplying positive   17 of 36              5 of 9                  -
verified concept count        -                     `concepts 9/9` in the log  (evidence, not inference)
```

**P1 MET:** positive supply falls to **19.839x**, against the pre-registered bar of <= 30.44x (at least 20% below 38.051x). It
actually fell **47.9%**. The falsifier did **not** fire, so the structural line stays open.
**P2 PASSED:** accuracy is **+0.18 points**, against a 2.0-point bar.

## Why this is the first encouraging result in the concept line

Three things moved the right way at once, for the first time:

1. **Cancellation nearly halved** (38.051x -> 19.839x).
2. **Every concept passes its audit** - 9 of 9, where the 36-concept layer manages 25 of 36. A concept layer whose units all mean
   what their names say is the actual goal, and this is the first configuration that gets there.
3. **Accuracy did not pay for it** (+0.18).

## Three caveats that travel with those numbers

1. **Perplexity is 33% worse** (956.79 -> 1275.33) while accuracy is slightly better. Those two signals disagree, and a model that
   proposes the right class while being far less confident is not simply "better". The honest summary is: **accuracy flat, calibration
   worse**.
2. **The concentration test is VACUOUS at 9 concepts.** Its bar is "top-8 suppliers carry >= 80% of positive supply", and with only 5
   positive suppliers the top-8 set is the whole layer - the tool printed "MET" and that MET means nothing. The bar does not scale
   with layer size; it must be restated as a fraction of the layer (e.g. top half) before it can be quoted at any count.
3. **Count and breadth are confounded.** The 9-concept catalogue also has broader definitions, so this is "fewer, wider" as the
   falsifier specified, **not** a pure count effect.

## Next

The question is no longer whether the concept layer can be made honest - at 9 concepts it is. It is whether the **good** shape is
narrow concepts WITH the direct path, since the 36-concept direct-path arm reached 12.18% while every bottlenecked arm sits near 7.4%.
That arm needs no new code either.
