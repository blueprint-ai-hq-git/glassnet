

---

# IT IS NOT THE HEAD: THE 36 CONCEPTS DO NOT CONTAIN THE ANSWER AT ALL (2026-11-22 15:42)

The previous measurement showed the readable route reproduces the model's answer only 5.5-18.0% of the time. Two repairs were possible:

1. **the head is wired wrong** - the information is in the concepts, and the answer head just fails to read it;
2. **the layer is too thin** - the 36 concepts do not contain the answer, and no head can recover what is not there.

The test is generous to option 1: **fit a brand-new linear readout from the 36 concepts to the model's own logits**, on half the windows,
and measure agreement on the other half. It is allowed to use the model's own logits as its target and least squares to find the best
linear map. If even that fails, no head design can rescue the readable path.

```
windows 1200 (fit 600, test 600) | concepts 36 | vocabulary 12,002
the model's own concept head agrees with its own answer:        23.2%
a NEW FITTED linear readout from the 36 concepts reaches:       24.0%            (top-10 hit 63.7%)
P1 fitted readout >= 70%: NOT MET
FALSIFIER FIRED
OUTCOME: LAYER_LACKS_ANSWER
```

## What this settles

**A best-possible linear readout of the 36 concept activations recovers the model's own top token 24.0% of the time - statistically
indistinguishable from the head the model already has (23.2%).** The extra 0.8 points is not a repair, it is noise.

So:

- **The repair is NOT a head repair.** Re-wiring the answer head cannot fix this, because the information is not in the concepts.
- **The readable middle layer must itself change** - it needs to be wider or built differently, so that its activations actually
  determine the next token instead of describing the sentence while the answer travels elsewhere.
- **The readable path's ceiling right now is ~24%**, and it is a property of the layer, not of the readout.

## Why this is the most useful result of the day, even though it is negative

It removes an entire class of wasted work. Before this measurement, the obvious next step was to rebuild the answer head - a day of work
that would have produced a few points of agreement and looked like progress. **The falsifier cost twelve seconds and saved it.**

It also makes the target unambiguous. The gate for the readable layer is no longer a soft one ("concepts should matter"). It is:

> run `tools/direct_ablation.py`; the **concept route must reproduce the model's own answer at 100%**, and
> `tools/concept_holds_answer.py` must show a fitted readout **at or above 90%** before a head change is even attempted.

Both numbers are produced by scripts that already exist, on data already on disk, in seconds. The readable layer now has a real gate
instead of a story.
