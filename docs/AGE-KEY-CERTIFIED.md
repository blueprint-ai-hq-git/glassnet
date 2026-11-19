

---

# THE AGE KEY: THE WINDOW BANK CAN NOW ADDRESS SLOTS BY AGE, AND ITS GRADIENT IS CERTIFIED (2026-11-19 15:25)

The measured defect: the readable window bank's read had **no position or age information**, so the only way to select a slot was content similarity. With a
full 16-slot window, a uniform read returns a vector whose nearest real token is one of the 16 only **5.0%** of the time (measured on a real trained embedding
matrix, V=12002 D=128). Retention was never the problem - slots are replaced wholly and nothing decays - so addressing was the whole limitation.

## The repair

Slot `j` at step `t` holds data written at step `t - age`, where `age = (t - j) mod K`. That age is identical for every sample in the batch, so the fix stays
exactly batchable:

```python
age = (t - np.arange(K)) % K
s = np.einsum("bkh,bh->bk", Km + p["age_emb"][age], q) / np.sqrt(H)
```

and in the backward pass:

```python
np.add.at(g["age_emb"], age, (ds[:, :, None] * q[:, None, :] / np.sqrt(H)).sum(axis=0))
dq = np.einsum("bkh,bk->bh", Km_prev + p["age_emb"][age], ds) / np.sqrt(H)
```

It is **opt-in**: everything activates only when `"age_emb"` is present in the parameter dict, so no existing checkpoint, script or running job changes.

## The gradient gate

No new gradient path is trusted here without central finite differences - that rule has already caught a missing direct-write term in `lens/tokmem.py` and a
wrong outer-product axis in the batched bank.

```
AGE-KEY FINITE-DIFFERENCE GATE  (H=4 D=3 V=5 K=3 T=5 B=3)
  parameters checked: 71  (includes age_emb (3, 4) and every path the key changes)
  worst absolute error: 1.713e-09   at Wk[3, 1]
  worst relative error: 4.157e-07
  GATE: PASS
  absence of age_emb gives identical forward: max |diff| 9.849e-02 (expected NONZERO: the key is active)
  zeroed age_emb gives identical forward:      max |diff| 0.000e+00 PASS (a zero key is a no-op)
```

Three things this establishes, each measured rather than asserted:

1. **The gradient is exact** (1.7e-09 absolute over all 71 parameters, including `age_emb` itself and the Wq/Wk/Wo/emb paths whose backward now differs).
2. **The key is genuinely active** - removing `age_emb` changes the forward by 9.8e-02, so the test is not passing on a no-op.
3. **It is safe for existing artefacts** - zeroing `age_emb` reproduces the old forward bit-exactly (0.000e+00), so a checkpoint saved without the key loads
   into the new code and behaves identically.

## What is NOT yet established

Whether it helps. The gate proves the mechanism is implemented correctly, not that it improves recall or next-token accuracy. That needs a training run, and
it must be judged on an instrument that can see the difference - not on next-token accuracy, whose seed noise (2.89) is 10x the effects this project has been
chasing. The delayed-recall probe is now a fair test of the window bank **for the first time**, because the architecture can express positional addressing.

## Next

Train the window bank with the age key on the delayed-recall task, at delays 1, 2, 4, 8 and 16, against a control with the channel removed. Predictions to be
written before launch: the age-key arm clears 50% at delay 8 (where the architecture previously had no way to address), the control stays at chance (1/32 =
3.12%), and recall degrades with delay. Falsifier: no better than the pre-age-key bank at every delay means addressing was not the limiter either, and the next
suspect is the read's linearity.
