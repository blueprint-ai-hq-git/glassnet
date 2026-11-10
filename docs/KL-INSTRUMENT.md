# The KL instrument is sound, and the trained readout is no better than random noise

Before trusting any more "input-flat" verdicts, the measuring instrument itself was tested
(`tools/kl_instrument_check.py`, checkpoint `scan8_ep750.npz`). It deliberately damages the readout in known
ways and reports what the metric does.

```
baseline KL (six words, model as trained)   : 0.0151 nats
control, readout unchanged                  : 0.0151   (identical, so the metric has no noise floor)

real readout scaled x0.25                   : 0.0056
real readout scaled x0.50                   : 0.0072
real readout scaled x2.00                   : 0.0470
real readout scaled x8.00                   : 0.5989

readout replaced by random noise, sd x0.1   : 0.0051
readout replaced by random noise, sd x0.5   : 0.0072
readout replaced by random noise, sd x1.0   : 0.0152
readout replaced by random noise, sd x4.0   : 0.1753

readout borrowed from scan8_ep500           : 0.0151
```

## Verdict 1: the instrument is trustworthy
It responds monotonically to deliberate damage (0.0056 at x0.25 through 0.599 at x8), and an unchanged
readout reproduces the baseline exactly. It is not blind, it has no noise floor, and it can register a large
change. So every earlier KL verdict stands as measured.

## Verdict 2: the trained readout is indistinguishable from random noise
A readout of **random numbers at the same standard deviation as the trained one gives KL 0.0152**, against the
trained readout's **0.0151**. Training did not make the readout any more sensitive to the input than not
training it. That is the cleanest statement yet of why E2 fails while E1 passes: the states differ, and the
answer layer throws the difference away as thoroughly as noise would.

It also explains why every previous readout change was a dead end. The question was never "which readout
tweak helps" - it is "why does this readout learn nothing about the state at all", with the state-side
explanations now all measured and eliminated (capacity, bias, frequency prior, training length, and state
collapse via the decorrelation penalty).
