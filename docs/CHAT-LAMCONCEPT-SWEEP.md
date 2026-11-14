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

## Fourth arm: lam-concept 1.2 finishes the curve, and it holds up under the corrected verdict

Predictions were pre-registered in `lens/runs/CHAT-LAM12-PREDICTIONS.txt`. Result in `lens/runs/CHAT-LAM12.txt`,
log `logs/exposure/lamconc_1p2.log`.

```
lam      old pass   strict pass   median AUROC   mean AUROC   fires > 0.90   VAL acc
0.1        19/36      20/36         0.56490       0.58931         ?            18.86%
0.3        25/36      18/36         0.61657       0.62950         ?            18.88%
0.6        31/36      21/36         0.67186       0.66788        11            18.89%
1.2        36/36      30/36         0.76235       0.75623         6            18.92%
```

- **P1 HOLDS**: 36 of 36 pass the trainer's own audit at 1.2, above the predicted 33.
- **P2 FAILED**: I predicted accuracy would finally pay, dropping to 18.59% or below. It **rose to 18.92%**, the highest of all four
  arms. Concept pressure is still free at 1.2 on this scale.
- **Falsifier NOT TRIGGERED**: honesty did not saturate at 31/36.
- **The strict verdict keeps climbing too**: 18 -> 21 -> 30 of 36, and median AUROC **0.61657 -> 0.67186 -> 0.76235** for lam
  0.3 -> 0.6 -> 1.2.
- **The near-constant problem HALVES**: concepts firing on more than 90% of windows fall from **11 to 6**, so the very property that
  made the old 0.55 bar misleading is itself improved by pressure, not worsened.

So the corrected bottom line, with every caveat intact: **on 2-epoch d512 chat arms, raising `--lam-concept` from the 0.3 default to 1.2
takes own-name honesty from 18 to 30 of 36 under a strict instrument and from 0.617 to 0.762 median AUROC, while validation accuracy
does not fall at all.** Whether that survives full training (9 epochs, d1024) and more seeds is the next question and is not claimed.

## RETRACTION the same night: the whole sweep measured an under-trained regime, and the shipped config already passes

Before spending an hour of GPU confirming lam 1.2 at full training, I checked the audit logs of the six **full-length runs already on
disk**. They contain the same per-concept audit, so the comparison was free.

```
log                    epochs  median AUROC   mean AUROC   old pass   STRICT pass   fires > 0.90
d1024_e9_s0              9       0.97094       0.96649      36/36       36/36            0
d1024_e9_s1              9       0.97267       0.96618      36/36       36/36            0
d1024_e9_s2              9       0.96756       0.96616      36/36       36/36            0
d512_e9_s0               9       0.87449       0.87846      36/36       36/36            0
d512_e9_s1               9       0.87064       0.87390      36/36       36/36            0
d512_e9_s2               9       0.87398       0.87782      36/36       36/36            0
dropunk_d1024_s0         9       0.97111       0.96547      36/36       36/36            0
lamconc_0p1              2       0.56490       0.58931      19/36       20/36            0
lamconc_0p3              2       0.61657       0.62950      25/36       18/36           16
lamconc_0p6              2       0.67186       0.66788      31/36       21/36           16
lamconc_1p2              2       0.76235       0.75623      36/36       30/36            6
```

**Every full-length run already passes 36 of 36 under the STRICT verdict, with a median AUROC of 0.87-0.97 and ZERO near-constant
concepts. Every short arm is far below that, whatever lam says.**

So the following claims from this document are **WITHDRAWN**:

1. "The default `--lam-concept 0.3` is leaving 6 concepts per run on honesty for nothing." **False at real training scale** - the
   default already delivers 36/36 strict at 9 epochs. Nothing needs changing.
2. "Raising lam takes honesty from 18 to 30 of 36." True only inside the 2-epoch regime. It is **not** a knob for the shipped model.
3. "11 of 36 concepts fire on more than 90% of windows" (the structural finding). **An artefact of under-training** - all six
   full-length runs have **zero** such concepts. The near-constant epidemic appears at 2 epochs and disappears by 9.

What actually survives:

- The **instrument flaw is real and now fixed** (`lens/api.py` strict verdict + `tools/test_strict_audit.py` 7/7): the 0.55 bar and
  2% fire floor do punish a well-detected rare concept and do reward a near-constant one. On full-training checkpoints the flaw is
  harmless because nothing is near-constant, but the fix is still correct.
- At **low compute**, concept pressure does improve concept ranking (median 0.565 -> 0.762 across lam 0.1 -> 1.2). That is a real
  measurement of early training, and it is the only thing the four arms ever showed.

The lesson, recorded because this is the third time it has bitten: **a short arm answers a question about short training, never about
the shipped model.** The check that caught it was free - reading the audit numbers out of logs that were already on disk - and it
should have come BEFORE the sweep launched, not after.

## The question that is actually open now

Full training passes 36/36 strict with median AUROC 0.97, which is suspicious for a *different* reason: the concept auditors score each
concept neuron against **the same deterministic label function the neuron was trained to predict**. A high AUROC there partly measures
"it learned its own training target", not "the name means what a human means". The honest follow-up is therefore not another knob sweep
but an **independent-label check**: score the concepts against labels produced by a different rule (different token lists or a
hand-checked sample), and a **causal check** like the image line's explain() test, where removing the top named rules must move the
logit far more than removing random rules.
