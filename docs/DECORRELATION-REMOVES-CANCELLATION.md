

---

# DECORRELATION REMOVES 73.5% OF THE CANCELLATION, FOR FREE (2026-11-20 12:56)

## The result, and the naming collision that nearly hid it

My script contained **two different predictions both labelled "P1"**, and the log printed only the tool's one:

- **the ARM's P1** (pre-registered in `run_decorr_cancel.sh`): the decorrelated arm's positive supply falls by at least **20% relative**
  to its paired control. **THIS IS THE ONE THAT DECIDES THE EXPERIMENT.**
- **the TOOL's P1** (printed by `concept_rank.py`): the top-8 positive suppliers carry at least 80% of all positive supply. That is a
  concentration test, and it printed "NOT MET" in both arms.

The log shows "P1 verdict: NOT MET" twice, which reads as a failed experiment. It is not - it is my naming collision. Recorded here
rather than renamed away, and the script's grep now has to be read against this entry.

## Measured, 100,000 windows, 96 usable windows per checkpoint, identical except one switch

```
                        control (lam_decorr 0)   decorrelated (0.2)     delta
held-out accuracy       7.31%                    7.25%                  -0.06
perplexity              953.87                   955.09                 +1.22
concepts passing audit  25 of 36                 25 of 36               0
POSITIVE SUPPLY         236.039x the margin      62.600x the margin     -73.5%
concepts positive       19 of 36                 17 of 36               -2
top-8 share of positive 72.3%                   70.7%                  -1.6
reconstruct residual    2.556e-01                2.544e-01              closes
```

- **The arm's P1 is MET, and by a wide margin:** positive supply falls **236.039x -> 62.600x**, a **73.5% reduction** against a 20%
  bar. Decorrelation is a real lever on cancellation.
- **P2 PASSED:** **-0.06 points** of accuracy (7.31 -> 7.25) against a 2.0-point bar. The cancellation removal is essentially free.
- **The tool's concentration test remains NOT MET:** the top-8 suppliers carry only **70.7%** of positive supply, so attribution is
  still spread across ~17 concepts. Cancellation fell but did not become concentration.

## The control reproduces the earlier arm exactly

Control here (953.87 ppl, 7.31%, 236.039x) reproduces the independent `arch_direct_off` 100k run (953.87 ppl, 7.31%, 236.032x) to
three decimal places on perplexity and accuracy. The pathological cancellation is a **stable property of this architecture**, not a
one-off.

## The residual I refused to trust last wake, now resolved

`max |sum(contribution) - margin|` was **4.632** for the `--direct` arm and **2.556e-01** for the bottlenecked arm. The difference is
not an attribution error: the concept terms can only ever reconstruct `(h@W)[pred] - (h@W)[runner]`, so what remains is
`(b + direct)[pred] - (b + direct)[runner]` - **the class-dependent bias and the direct path**, which the split cannot supply by
construction. The `--direct` arm has the large residual precisely because that is where the direct path lives, which is the same
reason it cancels less. The **236x figure is measured on the arm whose decomposition closes to 0.256**, so it is trustworthy; the
`--direct` arm's 8.476x carries the caveat.

## What is next, dictated by this number

Cancellation fell 73.5% at `lam_decorr 0.2` with no accuracy cost. The open question is whether it keeps falling, and whether
concentration ever crosses the 80% bar, at stronger decorrelation - so the next arm sweeps 0.5 and 1.0 against this same control
with the same exact instrument.
