# FIX-01 — the two broken concepts. PRE-REGISTERED BEFORE RUNNING.

Date: 2026-10-07 14:35. Written before the run, per the standing rule.

## WHAT IS BROKEN

Decoding the model into words (`lens/scripts/decode_to_words.py`) showed the cause of a failure
that four audits had only ever reported as "AUROC ~0.50":

```
pronoun_subject     = -7.86*George +7.28*Let -7.25*kind +7.24*o' -7.07*Duke +7.06*KING +6.87*Give
is_a_dialogue_line  = -9.08*KING -5.71*OF -5.40*let -5.24*DUKE -5.08*HENRY -4.83*DUCHESS -4.73*Let
```

* `pronoun_subject` hunts CHARACTER NAMES, not he/she/they/we/I.
* `is_a_dialogue_line` has NEGATIVE weights on KING/DUKE/HENRY/DUCHESS -- the speaker labels that
  define it. It fires when they are absent.

## THE DIAGNOSIS

Both concepts have **~25% support** (0.2507 and 0.2319) and both are driven by the same signal:
the Shakespeare text is dialogue-heavy, so speaker labels and pronouns co-occur constantly. The
concept layer has exactly **9 slots for 9 concepts**, so there is no spare capacity for two
overlapping ones. The model appears to have split them into **plus and minus** -- one took the
positive direction, the other the negative -- and the loser degenerated into noise.

Supporting evidence from the earlier record: this pair also failed in Targets D, E, G2 and on the
rebuilt engine, always at AUROC ~0.50. And Target C, which had a much weaker concept layer, got
`pronoun_subject` to **0.9982** -- so the concept IS learnable, it is being squeezed here.

## THE ONE CHANGE

`lam_concept`: **0.1 -> 1.0**. Nothing else changes. Same data, same architecture, same seed,
same 60 epochs, same batch, same learning rate, same `lam_act=0.2`.

This makes the concept supervision 10x stronger relative to the language-modelling loss. If the
concepts are merely under-trained, stronger supervision should fix them.

## PREDICTION (written before the run)

**PREDICTION 1:** at `lam_concept=1.0`, BOTH `pronoun_subject` and `is_a_dialogue_line` reach
AUROC >= 0.55 and pass the audit, giving **9/9 concepts**.

**PREDICTION 2:** the readable model's perplexity gap vs the matched normal gets **WORSE** than
+12.281, because stronger concept supervision pulls the model further from pure language modelling.

**PREDICTION 3:** if PREDICTION 1 is WRONG -- the two still fail at 0.1x supervision strength --
then the cause is the **concept design**, not the training strength, and the fix is to redefine or
drop one of the two rather than to train harder.

Prediction 3 is the important one: it is written down so that a failure cannot be reinterpreted
afterwards as "needed more training" or "passed in spirit". **A failure is a result.**

## HOW IT IS MEASURED

Same audit as everywhere else, bars unchanged: AUROC >= 0.55, firing floor 2%, "at least half pass"
for the arm. Per-concept AUROC and firing rate reported for all nine. Coverage reported WITH the
audit, never alone.
