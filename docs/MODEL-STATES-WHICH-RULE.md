

---

# THE MODEL STATES WHICH RULE IT USED: 100.0% ON ALL THREE SEEDS (2026-11-22 15:39)

## The task

One cue token in the input says **which rule applies** - *take the larger digit* or *take the smaller digit* - and **it changes per
example**. The model must write three tokens:

```
[which rule it used] [position of the digit that rule picks] [that digit]
```

The first token cannot be a constant. It has to be read from the cue and reported: a concept the model states **about its own
reasoning**, not a label attached afterwards.

## Measured, 6,000 steps, 200 held-out examples each

```
seed 0:  reason 100.0%  | position 100.0%  | value 100.0%  | all three 100.0%
seed 1:  reason 100.0%  | position 100.0%  | value 100.0%  | all three 100.0%
seed 2:  reason 100.0%  | position 100.0%  | value 100.0%  | all three 100.0%
OUTCOME: MODEL_WRITES_ITS_OWN_REASON
```

## Established, and not yet established

**Established:** a model of this kind can write down **which rule it is following**, at 100%, when the rule is chosen per example and the
writing is part of the answer. Added to the day's other results, the readable route now carries the answer perfectly for: arithmetic with
carries (2-8 digits, add and subtract), a choice between one, two and three items, and **the rule being applied**.

**Not established:** stating the cue is not reporting the act. A model could parrot the cue while picking a digit the cue did not ask for,
and its "reason" would still look correct. The stronger claim - **the stated reason matches the digit it actually chose** - is queued as
`run_cue_honesty.sh` and already running, with a four-rule version (larger/smaller/first/last) and a wider-input version behind it.

## The honest state of the readable layer after today

It **fails completely** as 36 concepts chosen and read out by me (the concept route reproduces the model's own answer 5.5-18% of the time;
three readouts all land at or below the existing head; concept layers of width 8-72 never beat the baseline on real text).

It **succeeds completely** as tokens the model writes itself.

**The readable layer has to be written by the model, not attached by me.**
