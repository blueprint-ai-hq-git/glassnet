# ARCHITECTURE FLAWS, MEASURED, WITH THE REPAIR FOR EACH

Written 2026-11-19 at his instruction: "find architecture flaws and repair those and make it transparent for our model so it
can learn more and hold more". Every flaw below is stated with the measurement that found it, not with an opinion. Ranked by
how much it actually costs.

The shipped checkpoint measured here is `lens/runs/chat/td4x_r090_ep8000.npz`.

---

## FLAW 1 (the big one). The model is 98x more lookup table than computation.

```
parameter  what it is                size        share
Wa         answer table H x V        3,072,512   62.3%
emb        token embeddings V x D    1,536,256   31.1%
Wh         THE recurrent core           65,536    1.3%
Wx         input projection              32,768    0.7%
Wd         direct path                   16,384    0.3%
mem_*      memory bank (4 tensors)      196,616    4.0%
```

Wh is the only tensor that mixes information between tokens. **1.3% of the model does the talking; 93.4% of it is a table
mapping a state to a vocabulary entry.** A model shaped like this cannot reason, and no amount of training fixes a shape
problem - which is why held-out accuracy sat near 25% while the frequent-token band looked healthy.

**Repair:** tie the answer table to the embedding (`logits = h @ emb^T`) so the 3.07M parameter table disappears and the
tokens' own vectors do the answering, then spend the freed budget on **depth** - more recurrent layers, which is where
computation lives. This is not a trick to shrink the model; it is moving capacity from the table to the machine.

---

## FLAW 2. One recurrent layer means one step of composition.

The core is a single tanh recurrence. Any computation needing two sequential operations (this refers to that, which refers to
the earlier thing) has to happen inside one state update. Measured consequence: **effective context is 2 tokens** (accuracy
rises 9.27% at L=1 to 16.73% at L=2 and then stays flat to L=128).

**Repair:** 2-3 stacked recurrent layers, each with its own small readout. The per-layer readouts are also the transparency
prize - every layer's prediction becomes inspectable instead of only the last one.

---

## FLAW 3. tanh saturation silently caps how much the state can hold.

tanh compresses to (-1, 1). With a 256-wide state and no normalisation inside the recurrence, units that saturate stop
carrying information and keep receiving tiny gradients.

**Repair:** measure the saturated fraction first (a cheap instrument), then either a per-unit learnable temperature or a
different squashing function. Not changed blind - the saturation measurement decides.

---

## FLAW 4. 5.3% of training positions are <unk> and are discarded.

`<unk>` is 15,686,536 of 293,725,090 tokens (5.34%) and is masked out of both scoring and targets. So the model spends a
twentieth of its exposure learning to predict a token it is then forbidden to predict.

**Repair:** it is a dataset/vocabulary decision, not a model one: raise the vocabulary cut so the tail is rare rather than
unknown, or exclude <unk>-heavy documents. Cheaper than any architecture change and it is 5% of the corpus.

---

## FLAW 5. Readout normalisation is one gain per unit, so calibration is global.

Measured earlier: mean confidence 43.02% against accuracy 29.34%, ECE 0.1368, and one temperature takes ECE to 0.0521 with
accuracy unchanged. One global constant cannot fix a per-class overconfidence pattern.

**Repair:** per-class bias calibration on held-out data (V extra parameters, fitted after training). Cheap, measurable, and
it does not touch the architecture.

---

## FLAW 6. Gradient clipping at 1.0 is doing the work that a stable architecture should do.

The clip is active almost every step (logged gradient norms 2-6 against a clip of 1.0), which means the effective learning
rate is being set by the clipping, not by the schedule.

**Repair:** once depth and normalisation are in place, re-measure how often the clip binds. If it stops binding, the
architecture is the fix and the clip is a safety net; if it keeps binding, the recurrence is still unstable and that is the
next flaw to chase.

---

## ORDER OF WORK (one change at a time, each with its own gate)

1. Tied readout (FLAW 1) - biggest share of parameters, and it is a single line of maths. Gate: finite-difference check, then
   held-out accuracy against the untied model at the SAME core size.
2. Depth 2-3 with per-layer readouts (FLAW 2) - the actual capacity move, and the transparency payoff.
3. Measure saturation (FLAW 3) before changing anything.
4. Vocabulary cut (FLAW 4) - dataset side, can run in parallel.
5. Per-class calibration (FLAW 5) - after depth settles, since it is a post-hoc fit.


---

# FLAW 3 MEASURED AND DISMISSED; THE REAL MEMORY LIMITER FOUND (2026-11-19 08:47)

Flaw 3 said tanh saturation caps what the state can hold. Measured on `lens/runs/chat/td4x_r090_ep8000.npz` over 2,560 positions
with `tools/saturation_probe.py`:

```
|preactivation| > 2.0 (tanh' < 0.071):   0.00%
|preactivation| > 3.0 (tanh' < 0.010):   0.00%
|preactivation| mean 0.248 | 90th pct 0.518
```

**Zero saturation. Flaw 3 is not real, and the activation must not be touched.** That is worth knowing precisely because it was a
plausible-sounding flaw - the state is not sitting in tanh's flat region at all.

## But the same probe found the actual limiter

```
Wh spectral radius 0.9000 | largest singular value 2.1763 | mean singular value 0.6153
per-step memory operator diag(1 - tanh^2) @ Wh :  mean singular value 0.5613
```

The operator that carries information from one step to the next has a **mean singular value of 0.5613**, so the average direction
loses **44% of its perturbation every step**, and 0.5613^16 predicts **0.01% retention at 16 steps**. Meanwhile Wh's largest
singular value is **2.1763** against a spectral radius of 0.9000 - a strongly non-normal matrix where a few directions blow up
while the average shrinks.

**That is the mechanism behind the whole memory line failing**: the model is not short of saturation, capacity or training time -
the recurrence physically cannot carry a fact 16 steps because the average direction of its memory operator contracts by
nearly half each step. It also explains why radius clamping to 0.9 "worked" for the original forgetting while still leaving the
16-step gate failed at 4.73%.

## The repair this points to

Drive the memory operator's **mean** singular value toward 1.0, not the spectral radius. Radius is about the largest eigenvalue;
retention is about the average of the singular values of `diag(1-tanh^2) @ Wh`. A near-orthogonal or explicitly
singular-value-normalised recurrent matrix is the change, and the target is measurable: from 0.5613 toward 0.95+.
