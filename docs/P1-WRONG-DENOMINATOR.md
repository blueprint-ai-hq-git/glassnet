

---

# THE 3000-STEP ARMS NEVER CLEARED TRIVIAL: P1 WAS EVALUATED ON THE WRONG DENOMINATOR (2026-11-19 13:55)

This is the third correction today, and the one that matters most, because it invalidates the gate I used to declare the memory test "fair".

## The defect, in code

`lens/train_tokmem.py:119` computes accuracy with `<unk>` targets **excluded**:

```python
wts = (tg != UNK).astype(np.float64)
```

`<unk>` is **5.19% of the validation region** (25,951 of 500,000 tokens, measured). So the trainer's "HELD-OUT acc" is a **non-<unk> accuracy**, and an
absolute constant that includes `<unk>` positions is not the same denominator. `tools/eval_next_bands.py` scores the model on all positions (with
`lg[UNK] = -1e9`, so a `<unk>` target is a guaranteed miss) but lets the **constant guess `<unk>` and score on it**, because the guess comes from the
unfiltered target list.

So the P1 gate - "both arms clear the 14.70% constant" - compared **14.75% over 94.81% of positions** against **14.70% over 100% of positions**. That
is not a comparison. It was a favourable denominator difference of ~0.77 points, which is larger than the entire effect the experiment was built to
detect.

## The corrected measurement, one protocol, one draw, one pass

`tools/eval_any.py` now prints all three numbers from the same pass over the same windows:

```
checkpoint                   model(<unk>-excl)   constant(<unk>-excl)   verdict
win3kwinoff_K0_ep3000         12.97%              13.14%                 BELOW constant by 0.17
win3kwinon_K8_ep3000          12.80%              13.14%                 BELOW constant by 0.34
    (constant over ALL positions on the same draw: 14.70%)
```

- **P1 FALSE.** Neither arm clears the trivial constant once the denominators match. Both are **below** it.
- **P2 FALSE.** The window's paired delta on the independent instrument is **-0.17**, not +0.50.
- The budget ladder is therefore **wrong as previously published**: 3000 steps does NOT clear trivial. The trainer's 14.75% was never comparable to
  the 14.70% constant.

## What this does to the last four hours of conclusions

- The "fair test" I designed at 3000 steps **was not fair** - it was as under-trained as the 600 and 1200 step arms, and the only reason it looked
  fair was a denominator mismatch.
- Every trainer-protocol number ever compared to an absolute constant in this project has the same defect. The 6000-step base (21.74%) has not yet been
  re-scored on the corrected protocol and **must be before it is used as "clearly above trivial"**.
- The window-bank verdict returns to: **untested at a budget where the model beats trivial**, because no arm so far has qualified.

## The fix, so this cannot recur

`tools/eval_any.py` now reports, per checkpoint and per draw: the model's `<unk>`-excluded accuracy, the constant's `<unk>`-excluded accuracy, the
constant over all positions, and a literal `BEATS constant` / `BELOW constant` verdict. Any model-vs-constant claim now comes from one pass of one
instrument with one denominator.

## The rule

**A gate that compares two numbers must prove both numbers were computed on the same set of positions.** I wrote the P1 gate myself, ran it, declared
it met, published it, and only found the defect by auditing the instrument after a different discrepancy made me suspicious. The most dangerous
measurement error is the one that makes your hypothesis look alive.
