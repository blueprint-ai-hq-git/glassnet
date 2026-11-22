

---

# THE WIDTH QUESTION CANNOT BE ANSWERED BY PROBING - AND MY OWN DIAGNOSIS WAS WRONG TWICE IN TEN MINUTES (2026-11-22 16:06)

The goal was to turn today's negative result into a design number: *how wide must the readable layer be to carry the answer?* It was
supposed to be answerable by probing the state that feeds the concepts. It is not, and the two reasons are both instructive.

## Certified fact used

`window_input` (32,768 values) is **exactly** the input to the concept layer - checked, not assumed:
`sigmoid(window_input @ Wc + b)` reproduces the model's own `concepts` to **1.556695e-07**.

## Measurement

```
                     600 fitting windows      2,500 fitting windows
k = 36 dims                5.8%                     4.7%
k = 256 dims              11.0%                     8.1%
k = 512 dims              11.7%                    10.4%
36 concepts (reference)   24.0%                    24.0%
```

## Wrong diagnosis number one: "sample starvation"

A 512-dim probe scoring **below** a 36-dim probe looked impossible, so the instrument refused a verdict and blamed sample size. **The
rerun with 4x the fitting windows gave 10.4% instead of 11.7% - so it was never the sample size.** Good that the refusal happened; bad
that the reason attached to it was invented.

## Wrong diagnosis number two, and the real explanation

I had mis-stated my own model. **The concepts are not a linear function of the state** - they are `h = sigmoid(x @ Wc + b)`. A linear
probe on the raw state therefore cannot reproduce what a linear probe on the **sigmoided** concepts achieves. The non-linearity is the
entire difference, and it is why widening the probe did not help.

## What this settles

- **Layer width cannot be measured by probing at all.** The right experiment is to **train concept layers of different widths and ablate
  them** with `tools/direct_ablation.py`, which is the gate that already exists.
- **The 36-concept reference of 24.0% is now explained rather than mysterious:** a sigmoid layer of 36 units is a *useful non-linear
  feature map*, and that is precisely what it is doing - describing the sentence. It still does not decide the answer.
- **Two wrong diagnoses in ten minutes, both caught by re-running with a change:** the pattern from the whole day holds. The first
  explanation of an anomaly is usually a story; the second measurement is the evidence.
