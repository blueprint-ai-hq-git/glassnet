

---

# THE OVER-SUPPLY IS UNIVERSAL IN DIRECTION, NOT IN SIZE - AND MY OWN PREDICTION FAILED (2026-11-22 15:18)

Three chat checkpoints, the now-valid instrument, 60 windows each. All three accounting checks close to machine precision, so the
instrument is valid on every one:

```
checkpoint                      recon error   concepts positive   concept share   bias+direct   top-8 concentration
d1024_s1_best                    1.776e-15        23 of 36            993.4%        -893.4%          68.2%
d1024_e9_s1_best                 2.887e-15        22 of 36            143.9%         -43.9%          66.2%
direct1024_d512_s0_best          1.776e-15        23 of 36            217.7%        -117.7%          65.9%
```

## My prediction, written before the run, and what happened to it

- **P1 "the concept layer supplies more than 300% of the margin on ALL THREE": NOT MET.** One checkpoint is at **143.9%**.
- **FALSIFIER "any checkpoint where the concept share is <= 150%": FIRED** - 143.9% is below it.
- **P2 "the accounting closes at or below 1e-9 on all three": MET** (1.8e-15, 2.9e-15, 1.8e-15).

So the strong form of yesterday hour's claim is **withdrawn**: this is **not** a fixed architectural property of 993%.

## What IS stable across all three, and it is worth more

1. **The direction is universal.** On every checkpoint the concept layer supplies **more than the whole margin** (143.9% to 993.4%) and
   the bias-plus-direct path **removes** a large part of it (43.9% to 893.4%). The answer is always a residue of an over-supplied
   concept layer, never a direct sum of concepts. **The magnitude is a training outcome, the direction is structural.**
2. **The corrected supply split is stable: 22-23 of 36 concepts positive** on all three. The recorded "17 of 36" was the broken
   instrument; **22-23 is the real figure** and it does not depend on which model is measured.
3. **The concentration ceiling is stable: 68.2%, 66.2%, 65.9%** - three different models, all landing within **2.3 points of each
   other**, all just under the 80% bar. Three independent models converging on ~66-68% is not noise. **That looks like a ceiling of
   this shape, not a lucky miss.**

## The consequence for the repair

The thing to fix is **not** "make concepts bigger" and **not** "train longer". It is the **over-supply-and-cancel ratio**, which varies
5.5x between checkpoints (143.9% to 993.4%) while the resulting concentration stays pinned at ~66-68%. A layer whose concentration is
insensitive to a 5.5x change in its own share is not being read in a way that lets a small set of names decide the answer.

That statement is now bounded by three measured models rather than one, and by a falsifier that fired rather than a prediction that
flattered itself.
