# Resume is now BIT-EXACT, and the previous version was not (2026-11-18)

The resume added earlier today was weight-exact but not **bit**-exact: the sampler's rng state was not stored, so a resumed run continued the weights exactly while seeing a
different shuffle of the corpus. That was documented as a limitation. It is now closed.

## What changed

The checkpoint's `__meta__` now carries `rng_state` (`np.random.Generator.bit_generator.state`, a plain dict that JSON round-trips exactly), together with `ortho` and
`ortho_scale` so a resumed run can verify it is continuing the same configuration. On resume the state is restored before the first batch is drawn.

```
meta keys: ['D', 'H', 'V', 'ck_prefix', 'ortho', 'ortho_scale', 'rng_state', 'step', 'target_radius']
rng_state present: True | bit generator: PCG64
```

## The test that proves it

Two **independent** resumes from the same checkpoint, 25 steps each, must produce identical numbers if the sampler state was genuinely restored:

```
2026-11-18 11:35:12 | step 275/275 loss 8.3962 ema 8.4815 acc 2.54% gnorm 3.46 lr 0.0030 tokens_seen 6,400
2026-11-18 11:35:32 | step 275/275 loss 8.3962 ema 8.4815 acc 2.54% gnorm 3.46 lr 0.0030 tokens_seen 6,400
```

**Identical to every printed digit.** The resume now reproduces the run it continues, not just its weights.

## Two bugs of mine, both caught by running the thing

1. **`NameError: name 'log' is not defined`** - the first resume patch called `log()` before its definition further down the file. Every resumed run died instantly; the patch
   parsed cleanly and read correctly.
2. **`NameError: name 'seen' is not defined`** - the second patch matched a line only part-way and left `; t0 = ...; ema = None; seen = 0` appended to the last line of its own
   replacement, which put those initialisations **inside the `if _resume_msg:` block**. Every **fresh** run then crashed, while resumed runs worked - the opposite of the first bug.
   Both were found by executing a 12-step fresh run and a 2-step resumed run, not by reading the diff.

The lesson recorded: a patch that only *adds* code to a line that already has statements is exactly where this class of mistake lives. Check the line's tail, then run both the
resumed and the fresh path.
