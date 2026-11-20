

---

# DECORRELATION SATURATES AT 0.5, AND THE FALSIFIER FIRED (2026-11-20 13:02)

## The sweep, 100k windows, identical except the decorrelation weight

```
                                    accuracy   ppl      positive supply   P2 vs control
control            lam_decorr 0     7.31%      953.87   236.039x          -
d02                lam_decorr 0.2   7.25%      955.09    62.600x          -0.06   PASS
d05                lam_decorr 0.5   7.19%      956.79    38.051x          -0.12   PASS
d10                lam_decorr 1.0   7.13%      959.33    38.356x          -0.18   PASS
```

## Verdict against the pre-registration

- **P2 PASSED at every strength:** accuracy stays within 0.18 points of the control. Removing cancellation is cheap all the way up.
- **P1 FAILED:** I predicted supply(1.0) would be at least another 20% below supply(0.5). It is **+0.8% above** it (38.051x -> 38.356x).
- **FALSIFIER FIRED, as written:** "if supply(1.0) is NOT at least 20% below supply(0.5), decorrelation SATURATES at 0.2 and the
  residual cancellation is structural - the remedy then is FEWER, WIDER concepts, and this line closes rather than sweeping further."

The shape of the curve is the finding: **236.039x -> 62.600x -> 38.051x -> 38.356x**. The first two steps do the work (-73.5%, then
-39.2%); the third does **nothing**. The decorrelation line therefore **CLOSES at 0.5**, and the remaining **38x** cancellation is a
structural property of a 36-concept layer, not a tuning problem. No further weights will be swept - the strength family has already
been measured four ways elsewhere in this project and the pre-registration says close.

## The instrument disagreed with itself, and only one answer counts

For the `d10` arm the log printed **both**:

```
  CONCENTRATION (tool-side test): MET (>= 80%)                                    <- NON-additive path
  CONCENTRATION (tool-side test, not an arm prediction): NOT MET   (72.5%)        <- --fixed-runner path
```

The first line comes from the **re-computed-runner** path, which is the invalid instrument that produced -302% earlier: it cannot
be additive because ablation can change which class is runner-up. The second comes from the fixed-runner path. **Only the fixed-runner
number counts**, and it says concentration is NOT met (72.5% against an 80% bar). The tool now labels the invalid path as invalid in
its own output so this cannot be misread again.

## Honest state of the concept question after today

1. Share is not specificity (a bottleneck pushes share to 100% and the named-vs-random ratio DOWN).
2. The layer answers by cancellation - positive supply 236x the net margin.
3. Decorrelation removes most of it (236x -> 38x) at no accuracy cost, then saturates.
4. The remaining 38x is structural: **36 concepts is too many for this answer layer**, which is exactly what the falsifier named as
   the remedy - fewer, wider concepts.
5. Concentration never crosses 80% at any setting, so "a few concepts carry the answer" remains **false**.

The next structural experiment is therefore not another weight. It is a concept layer with fewer, wider units, and that needs a code
change this trainer does not have today (there is no concept-count knob: `--help` exposes only `--concept-target`,
`--lam-concept`, `--lam-act`, `--lam-decorr`, `--corr-cap-abs`). That change is queued with this constraint recorded rather than
approximated by a weight.
