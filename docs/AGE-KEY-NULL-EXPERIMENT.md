

---

# THE AGE-KEY END-TO-END ARM WAS A NULL EXPERIMENT: THE KEY WAS 1,117x TOO WEAK (2026-11-19 18:35)

The age-key arm (`win6kage`, 6000-step budget) was stopped at step 5025 by a forced shutdown. Before resuming it, I compared its training line against
its control (`win6kon` - same seed, same batch order, so the per-step lines are PAIRED):

```
matched steps: 201 (steps 25 .. 5025)
accuracy difference (agekey - control): mean -0.004 pts, sd 0.11, min -0.72, max +1.05
over steps >= 4000 (n=42):              mean -0.002 pts, sd 0.09
loss difference:                        mean -0.0018
late steps where agekey acc > control:  10 of 42
step  500: 5.31% vs 5.31%   step 2500: 8.51% vs 8.51%   step 5000: 13.97% vs 13.97%   (EXACTLY equal)
```

Two survivable explanations: the key does nothing, or the key **cannot** do anything. So I measured the key itself in the checkpoint:

```
keys saved: emb Wx Wh b Wd gain_ro Wp bank_Wq bank_Wk bank_Wo bank_age_emb   (the key IS saved)
age_emb |sum| 0.4544 at ep4750 -> 0.4617 at ep5000      (it IS training)
age_emb max|.| 0.002557, RMS 0.000357
key magnitude at write time (h @ Wk) RMS 0.3987
-> the age term is 0.1% of the key scale.  ratio = 1117x
```

**The age signal was 1,117 times smaller than the content signal it is added to**, so the attention could not separate ages by construction - it is added to the
same vector and then dot-producted with the query, so a 0.1% perturbation cannot change the softmax ordering. **The arm could not test the hypothesis. It is a
NULL EXPERIMENT, not a negative result, and it must not be reported as evidence about the age key.**

## Why the delayed-recall probe succeeded then

The probe also initialised `age_emb` at zero, and it produced +65.01 points at delay 8. The difference is scale pressure: in the probe (H=64, 3000 steps,
lr 0.05) the ONLY useful signal is age, so the key grows to where it matters; in the language model the content keys are informative and the learning rate is
sized for language modelling, so the key stayed negligible. **The probe's claim stands - the mechanism CAN express age addressing - and the end-to-end arm
could not exercise it.** Those are consistent statements.

## The repair, pre-registered

`AGE_INIT` now sets the key's initial RMS instead of zero, and the new arm uses **AGE_INIT = 0.40**, which is the measured key RMS (0.3987) rounded - so from
step 0 the age term is comparable to the content term. One change against the same control:

- **P1** the arm now DIFFERS from its control during training (mean paired accuracy difference above 0.5 points in magnitude somewhere) - i.e. the key is
  finally able to act. If training lines stay identical, the scale is still wrong and no endpoint number means anything.
- **P2** at 6000 steps it beats `win6kon` on the trainer line by >= 0.5 points.
- **P3** the independent evaluator gives the same sign.
- **FALSIFIER:** training lines diverge (P1 met) but neither instrument shows >= 0.5 points -> the key expresses age addressing and still does not help real
  text, and the memory line closes on measurement.

The boot job now runs `win6kage2` (AGE_INIT=0.40) instead of resuming the null arm, so the CPU goes to a test that can actually fail.
