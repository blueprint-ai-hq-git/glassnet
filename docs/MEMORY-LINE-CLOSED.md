

---

# THE MEMORY LINE CLOSES ON MEASUREMENT: THE AGE KEY WORKS ON THE PROBE AND BUYS NOTHING ON REAL TEXT (2026-11-19 20:15)

The decisive end-to-end test. `win6kage2` is the readable window bank with the age key at the **measured** scale (`AGE_INIT = 0.40`, key RMS 0.3987), 6000 steps,
seed 0, paired against `win6kon` - identical in every other respect, same batch order.

```
age key after training:     RMS 0.3907, max|.| 1.3418   (init 0.40)          -> the mechanism is properly armed this time
training lines, paired:     mean acc diff -0.006 pts, sd 0.20
                            64 of 240 steps identical, only 6 of 240 differ by more than 0.5 points
trainer line:               age-key 23.02%  vs control 22.88%   = +0.14        (P2 bar was +0.5)
independent evaluator:      age-key 20.56%  vs control 20.62%   = -0.06        (P3 required the same sign)
trainer's matched constant: model 23.02% | constant 14.47% -> BEATS by 8.55
independent constant:       model 20.56 / 20.62% | constant 13.14% -> BOTH BEAT
```

- **P1 at best marginal:** the training lines barely diverge (sd 0.20 points, and the earlier zero-init arm had sd 0.11) - so even at the correct scale the key
  does not materially change what the model learns.
- **P2 FAILED:** +0.14 against a +0.5 bar.
- **P3 FAILED:** the independent evaluator gives **-0.06**, the wrong sign.
- **The falsifier fired exactly as written:** "no gain on both instruments -> the key expresses age addressing and still does not help real text, and the memory
  line closes on measurement."

## Why this is a clean negative and not a null result

The previous arm could not test the hypothesis: its key was 1,117x too weak (RMS 0.000357 against a key scale of 0.3987). This arm fixes that - the key is at
**RMS 0.3907**, comparable to the content keys - and both arms **beat the trivial constant by 8.55 and 7.48 points**, so the budget qualified and the comparison
was meaningful. The mechanism is present, at scale, in a model that works, and it still buys nothing measurable for next-token prediction.

## What the whole memory line now says, in one paragraph

The readable window bank **is** readable (eight distinct real tokens at the end of training), its gradient is **exact** (4.334e-10), its batched path equals the
certified one (1e-16), it does not collapse (slot-to-slot cosine -0.081), and its missing piece - positional addressing - was found, implemented and **certified**
(1.713e-09), then shown to be decisive on the instrument built to see it: **delay-8 recall 20.61% -> 85.61%, three seeds, per-seed +62.1 to +69.5** against a
3.12% chance. And on the only objective that matters for the product - next-token prediction on real text - it is worth **+0.14 / -0.06 points at a qualified
budget**. The mechanism works; the objective does not reward it.

That last sentence is the finding, and it is consistent with the model's **2-token effective context**: an objective that never needs more than the preceding
token gives a recent-token buffer nothing to sell.

## My own error in this run, recorded

The boot script's evaluation step hardcoded the OLD checkpoint prefix (`win6kage`) while the arm trained as `win6kage2`, so the independent evaluation silently
did not run and the job still printed `AGEKEY-E2E COMPLETE`. I caught it reading the log (`FileNotFoundError`), re-ran the evaluation by hand, and the numbers
above come from that manual run. **A runner that prints COMPLETE after a failed step is the false-completion defect this project has a rule against** - the
script needs an exit-code check before it claims completion, and that is now the next tooling fix.

## Where the effort goes next

The band analysis says the headroom is in **ranks 50+**, which sits at 0.22% and 0.00% accuracy - so the model is a top-50 predictor with nothing beyond it. That
is a bigger and much cheaper target than memory: the diagnostic is done, the calibration route is closed, and the remaining candidate is a readout that is
actually asked to distinguish the long tail - with predictions written before launch.
