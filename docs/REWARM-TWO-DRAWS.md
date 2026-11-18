# The re-warm is bigger than published, and my 12-chunk probe is not fit for endpoint decisions (2026-11-18)

The second draw of the re-warm finished. Comparing it to seed 0 exposed a **direct conflict between two measuring instruments on identical checkpoints** - and the conflict has to be resolved
before any number from either is published.

## The numbers

```
                                       12-chunk probe   104-chunk (each trainer's own HELD-OUT)
base, 6000 steps                          25.89%              21.74%     (loss 9.4149)
seed 0, +1500 steps at lr 0.03            27.20%              26.95%     (loss 9.3506)
seed 1b, +1500 steps, new data order      24.47%              27.25%     (loss 9.3389)
gain, seed 0                               +1.31              +5.21
gain, seed 1b                                  -              +5.51
do the two draws agree within 2 points?   NO (-2.73)         YES (+0.30)
```

**The two protocols disagree in sign.** Both cannot be right.

## Which instrument is wrong, and how the project already knows

The 12-chunk probe scores **756 samples** (SE roughly 1.5 points); the trainer's own evaluation scores **6,656** (SE roughly 0.55). More importantly, this same probe **has now produced
two wrong-signed verdicts in one day**:

1. It said the symmetry-broken memory bank **cost** 0.66 points of accuracy; the 104-chunk numbers show it **gained** 0.19 (corrected in `MEMORY-BANK-ENDPOINT.md`).
2. It says the two re-warm draws disagree by 2.73 points; the 104-chunk numbers say they agree within 0.30.

An instrument that reports the opposite sign twice on the same kind of comparison is not a noise problem, it is **unfit for endpoint decisions**. The verdict here therefore rests on the
**104-chunk evaluation, which is the same code path for every arm and has 8.8x the samples**.

## What is now claimed, and what is withdrawn

- **Claimed:** on the authoritative 104-chunk protocol the re-warm lifts held-out accuracy from **21.74% to 26.95% (seed 0, +5.21)** and to **27.25% (seed 1b, +5.51)**, with the two draws
  agreeing to **0.30 points**. The pre-registered falsifier for the second draw ("held-out inside 25.24-26.01%") **does not fire** on this protocol - both draws are far above that band.
- **Withdrawn:** the previously published figure of **+1.31 points at 1.85 SE**. It is not wrong in sign, but it understates the effect by roughly 4x because it was measured on the
  12-chunk probe. The corrected figure is **+5.21 / +5.51 points across two draws**.
- **Also withdrawn:** the "+1.31" framing that presented this as marginal. On the reliable protocol it is not marginal.

## The rule this adds

A probe whose sample count is one-eighth of the trainer's own evaluation must not decide an endpoint. Every future arm report must state **which protocol produced each number**, and endpoint
decisions use the trainer's 104-chunk evaluation. The 12-chunk probe stays for cheap mid-run trend checks only, and any conclusion from it must be re-measured at 104 chunks before publication.
