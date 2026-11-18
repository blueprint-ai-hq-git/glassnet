# A correctness gate that could never pass, and what it was really saying (2026-11-18)

While wiring the memory channel I ran the module's own cross-implementation check for the first time:

```
check_against_single()  ->  (0.228831028804527, 'b')      tolerance is 1e-11
check_fd()              ->  (1.683581468681e-10, 'Wx[2, 0]')   PASSES
```

A 0.229 disagreement looks like a serious gradient bug in the path every chat arm trains through. It is not. **The check was broken, and it had been failing silently.**

## Finding the cause instead of assuming a bug

The two implementations were not even computing the same loss on the check's own data:

```
single-chunk loss 6.180194931868
batched      loss 1.890029695056      vocabulary size 7, so uniform is ln(7) = 1.9459
```

**6.180 > ln(7)** is the tell: a loss *above* uniform on a 7-token problem means something is paying for an impossible target. The single-chunk path masks vocabulary id 0 in the
logits (`lg[:, mask] = -1e9`), and `check_against_single` drew its test tokens from `0..V-1`, so **the masked id appeared as a target about one time in seven**. The gap is exactly
`-log(1e-12)/6 * 1 ≈ 4.6`, and `1.890 + 4.6 ≈ 6.18`.

The batched `train_step` excludes such positions. So the two paths differ **deliberately** on a masked target, and the check was drawing precisely the one token whose handling
differs.

## The proof, and the fix

Draw the test tokens from `1..V-1` so the masked id never appears as a target:

```
single loss 2.085672447526 | batched loss 2.085672447526 | difference 0.000e+00
worst gradient difference: 1.110e-16 at 'b'
```

**The two implementations agree exactly.** The batched path - the one every chat arm uses - was never wrong; `check_fd` had already certified its analytic gradients against finite
differences at 1.68e-10. `check_against_single` now draws from `1..V-1` and documents why, and it passes for the right reason instead of failing for a wrong one.

## Why this mattered more than the next experiment

A gate that can never pass is worse than no gate: it teaches everyone to ignore the gate. This project leans on its gates for every claim it publishes, so a permanently-red check is
not a harmless wart - it is an erosion of the thing that makes the numbers trustworthy. The same scrutiny applied to a *passing* number (the eval protocol that was too small, the probe
that printed a hardcoded retention) applies to a *failing* one: **audit the instrument before blaming the model.**
