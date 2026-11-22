

---

# "DOES IT MEAN WHAT ITS LABEL SAYS?" — 12 OF 14 CONCEPTS HONEST, AND ONE REAL LIAR FOUND (2026-11-22 14:02)

The readable middle layer's recorded failure is that in the chat model only 17 of 36 concepts supply their own label while 19
oppose it. Before trying to repair that, I certified the instrument that decides "does this neuron mean its name", because
`lens/concepts.py` already records three earlier occasions where **the audit itself was the broken thing**.

## The instrument is tested by construction, and the test caught a real flaw in my own audit

Four neurons are planted with verdicts known in advance. The first version of the audit **failed its own test**:

```
HONEST_COPY (a PERFECT copy of label 0)   own=1.00 redundant=0.92 unique=+0.07  -> DISHONEST   *** WRONG ***
```

A perfect copy called dishonest, because `ink_amount` is 92% predictable from the other concepts — so **no** neuron could ever be
0.10 more informative than them. The bar was absolute where the ceiling is not. Fixed to a **fraction of the label's own
predictable information** (captures > half of what makes its label unique), plus a hard requirement that the neuron's strongest
match is its own name. After the fix:

```
HONEST_COPY (is label 0, named label 0)   own=1.00 frac=0.96 -> HONEST     CORRECT
IMPOSTOR   (is label 1, named label 0)    own=0.02 frac=-10.66 -> DISHONEST CORRECT   <- secretly detects something else
MIXED      (0.6 label0 + 0.4 label1)      own=0.94 frac=0.31 -> DISHONEST CORRECT
NOISE      (random)                       own=0.01 frac=-10.76 -> DISHONEST CORRECT
```

**P1 MET: all four correct.** The **impostor is the case the project brief demanded to see caught** — a neuron that secretly
tracks a different concept than its name claims — and the audit catches it.

## The real result: fitted on 5,000 images, audited on a DIFFERENT 5,000

Exact least-squares concept decoders (hand written; the first attempt used gradient descent and reported `own=0.00` for every
neuron, meaning it measured my learning rate and not the concepts, so the learning-rate knob was removed entirely):

```
  named concept             own     redund   frac  actually tracks        verdict
  ink_amount               1.00       0.92   1.00  ink_amount             HONEST
  quad_UL_ink              0.95       0.87   0.58  quad_UL_ink            HONEST
  col_ink_std              1.00       0.48   1.00  col_ink_std            HONEST
  corner_ink               0.79       0.75   0.17  corner_ink             weak
  row_ink_std              0.69       0.74  -0.19  unnamed_13             WRONG NAME
  ink_entropy              1.00       0.78   1.00  ink_entropy            HONEST
  ink_cols_nonzero         1.00       0.87   1.00  ink_cols_nonzero       HONEST
  grad_ratio               1.00       0.92   1.00  grad_ratio             HONEST
  symmetry_lr              1.00       0.90   1.00  symmetry_lr            HONEST
  ink_rows_nonzero         1.00       0.91   1.00  ink_rows_nonzero       HONEST
  unnamed_10/11/12/13      1.00/1.00/0.92/0.93     own name               HONEST (4)
RESULT 12 of 14 concept neurons mean their own name
```

**Two honest failures, both worth having:**
- **`corner_ink` is `weak`** — it does match its own name best (own=0.79) but captures only **17%** of what makes it unique, so
  it mostly repeats other concepts. A concept that cannot be distinguished is not a concept you can read off.
- **`row_ink_std` is `WRONG NAME`** — it matches `unnamed_13` better than its own label (own 0.69 against a higher cross
  correlation). **This is a real neuron secretly tracking something else**, found on real data rather than in a toy.

## What this changes

- The concept idea is **sound when the concept is measured directly**: 12 of 14 mean their name on images the decoders never saw.
- The chat model's 17-supply/19-oppose cancellation is therefore **not a flaw of concepts as such** — it is specific to how
  concepts are attached inside the language model, which remains the last broken layer.
- `row_ink_std` must be **dropped or renamed**, by this project's own rule: a concept that cannot be label-honest is dropped, never
  forced.
