# Prepared, not launched: a measured concept-widening design (36 -> 60 concepts)

Date: 2026-11-14 20:30. **Nothing was trained for this document.** It is the preparation that makes the pending widening decision
actionable, and it waits on the owner because adding concepts changes the definitions every audit compares against.

## Why widening is the last lever

Everything else has been measured and come back negative on the readable chat model:

```
lever                                  result
hidden width (d512 -> d1024)           readable VAL 19.95% -> 19.95%; diversity 44 -> 49; concept names +0.006
post-hoc readout repair (4 variants)   all fail, 3 worse than doing nothing (no_unk 3.15%, centred 0.60%)
bias (de-centre / exclude <unk>)       not the cause: bias agreement 19.5%, 2.55% alone
more training (9 -> 18 epochs)         helps (VAL 19.95 -> 21.48, tokens 49 -> 78) but does not close diversity or name gaps
concept supervision pressure            raises own-label scores on under-trained arms, does nothing at full length
```

The constraint that explains all of it: **13-18 of 36 concepts fire on a median window**, so the code is half-saturated and the
36 -> 12,002 projection learns only ~78 output modes, and each neuron is pressed into service for information unrelated to its name.

## Batch 1: 26 candidate concepts, measured against the real corpus

Each candidate is a token list in the same frozen form the existing concepts use. Support was measured on 20,000 held-out windows of
the full 294M-token corpus (the same floor the trainer applies is 0.005):

```
candidate        toks  support   verdict              candidate        toks  support   verdict
code_refs           8  0.2623   KEEP                  money_amt           7  0.0180   KEEP
modality            8  0.2261   KEEP                  certainty           8  0.0175   KEEP
help_ask            8  0.0850   KEEP                  motion              8  0.0170   KEEP
sequence            8  0.0846   KEEP                  gratitude           6  0.0149   KEEP
commands            8  0.0663   KEEP                  weekday             8  0.0147   KEEP
comparison          8  0.0621   KEEP                  time_hour           8  0.0144   KEEP
numbers_word        8  0.0600   KEEP                  sizes               8  0.0120   KEEP
emotion_pos         8  0.0459   KEEP                  shape               8  0.0083   KEEP
thinking            7  0.0365   KEEP                  tech_gear           8  0.0075   KEEP
cause               8  0.0295   KEEP                  greeting            7  0.0073   KEEP
frequency           8  0.0211   KEEP                  apology             6  0.0064   KEEP
senses              8  0.0200   KEEP                  emotion_neg         8  0.0016   DROP
purpose             6  0.0185   KEEP                  weather_sky         8  0.0016   DROP
```

**24 of 26 survive the floor, which takes the concept set from 36 to 60.** Full table `lens/runs/CONCEPT-WIDENING-CANDIDATES.txt`.
Two candidates fail the floor on this corpus (`emotion_neg`, `weather_sky` at 0.0016) and are dropped rather than forced - the
project's own rule is that a concept which cannot be label-honest is dropped, never propped up.

## Predictions for the widening arm, written BEFORE it runs

To be run as: the readable (no-direct) configuration at d1024, same corpus/split/seed, with the 60-concept set, against the existing
36-concept readable arm as the control.

- **W1** diversity improves materially: **more than 110 distinct tokens** on the 2,000-position metric, against 78 for the 36-concept
  18-epoch arm.
- **W2** concept saturation falls: median concepts firing drops **below 13 of 60** (it is 13 of 36 now), because there are more
  concepts to share the predictive load.
- **W3** concept name honesty improves on the stricter instrument: held-out synonym median **above 0.60** (0.5676 now, against the
  direct-path model's 0.7178).
- **W4** accuracy does not regress by more than 0.5 points: final VAL stays **at or above 20.98%** (21.48% now).
- **FALSIFIER**: if diversity stays under 90 tokens AND held-out concept honesty stays under 0.58, then 60 concepts is not enough
  either and the readable design needs a different capacity mechanism than "more named concepts" - which would be a finding about the
  architecture, not about the size of the concept list.

## What this design deliberately does not do

- **It does not change any of the 36 existing concept definitions**, so results remain comparable with everything measured tonight.
- **It does not touch the direct pathway, the concept loss weight, or the readout.** One change at a time, per the project rule.
- **It does not launch anything.** Adding concepts to the trainer is a definition change and the owner's call.
