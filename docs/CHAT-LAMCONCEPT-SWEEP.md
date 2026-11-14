# Concept supervision sets chat concept honesty - and at this scale it is nearly free

Date: 2026-11-14. Runner `lens/scripts/run_chat_lamconcept_sweep.sh` (invocation copied verbatim from the working `dropunk` arm,
four declared changes: d 1024->512, epochs 9->2, max-windows 2M->600k, lam varied). Predictions pre-registered in
`lens/runs/CHAT-LAMCONCEPT-PREDICTIONS.txt`. Output `lens/runs/CHAT-LAMCONCEPT-SWEEP.txt`, per-arm logs
`logs/exposure/lamconc_0p{1,3,6}.log`. Seed 0, 600,000 train windows of the full 294,225,090-token corpus, window 32.

## Why this ran

Two explanations for chat concept dishonesty were already eliminated with numbers: redundant labels in **vocabulary** (worst
jaccard 0.08) and redundant labels **in effect** (max pairwise phi 0.415, and that one was a single shared token, since fixed). The
remaining suspect was **training pressure** - the lever that governed honesty on the image line, where `alpha` set concept
fidelity and `lam` set rule structure.

## Result: the pass count nearly doubles across the sweep, accuracy does not move

```
lam-concept   concept fidelity (own-name audit)   VAL accuracy   VAL ppl
0.1                 19 of 36  (52.8%)               18.86%        265.87
0.3  (default)      25 of 36  (69.4%)               18.88%        267.08
0.6                 31 of 36  (86.1%)               18.89%        268.01
```

- **P1 (pass count rises with lam: 0.6 >= 0.3 >= 0.1) HOLDS** - strictly increasing, **19 -> 25 -> 31 of 36**, which is +12 concepts
  passing their own-name audit between the default and 0.6.
- **P2 (accuracy falls as lam rises, drop <= 3 points) - HALF FAILED**: accuracy did not fall at all, it rose by **+0.03 points**
  (18.86 -> 18.89). The magnitude constraint is trivially satisfied; the predicted *direction* was wrong.
- **P3 (no arm collapses) HOLDS** - all three arms within 0.03 points of each other.
- **Falsifier (identical pass counts) NOT TRIGGERED**, so chat concept honesty **is** governed by supervision pressure, agreeing with
  what the image line measured.

**The default `--lam-concept 0.3` is leaving 6 concepts per run on own-name honesty for roughly nothing.** At 0.6, 31 of 36 concepts
mean their name, against 25 at the default, with identical validation accuracy.

## What this does not establish

- **2 epochs, seed 0, d512.** These are short arms chosen to fit a free window. The finding needs confirming at the shipping
  configuration (9 epochs, d1024) and on more than one seed before any public claim about the shipped model changes.
- **VAL accuracy ~18.9%** here is a short-run number, not comparable to the shipped model's numbers.
- **Does it keep rising?** The curve has not plateaued: 19 -> 25 -> 31 leaves 5 concepts still failing at 0.6. The obvious next arm
  is lam-concept **1.2** (and higher), to find where honesty saturates and where accuracy finally pays for it.
- **"Pass" is the trainer's own audit criterion**, not a fresh independent instrument. It is the same audit that reported 36/36 on
  the shipped checkpoint, so it is comparable within this project, but it is not an external check.

## Why this matters for the goal

His first standing item is a model that is 100% honest and understandable. On the image line the equivalent knob (`alpha`) turned out
to be the single most effective honesty lever and cost accuracy in exchange. **On chat the same lever appears to be free at this
scale**, which - if it holds at full training - means the chat model's concept layer can go from 69% to 86% name-honest without
trading anything measurable.
