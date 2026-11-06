# Roadmap and release plan

## Release schedule

| Version | Date | Content | State |
|---|---|---|---|
| **v0.1.0 - source** | **2026-11-10** | Engine, model, tests, all results and failures, licence, docs | repo private, ready to flip public |
| **v0.1.1 - weights** | 2026-11-12 | 37.37% checkpoint exported to `.safetensors`, gated on HuggingFace | needs export |
| **v0.2.0** | 2026-11-17 | Residual normalisation fix, re-test vs the 5.53% baseline, GPU comparison | planned |
| **v0.3.0** | 2026-11-24 | Proper multi-head attention (after `reshape` gradient is settled), audited concept coverage as a printed number | planned |
| **v0.4.0** | December | Safety layer trained in, not bolted on; coherent-conversation milestone | planned |

Dates are commitments to publish *something measured*, never to a specific accuracy. If a version ships
without a number, it does not ship.

## Why release early, at 37%

Because the interesting part is not the score. The engine is verifiable, the failures are documented, and
every claim traces to a file. Releasing at 37% with honest numbers lets people test the *method* now -
waiting until it looks impressive would mean shipping something nobody can check.

## Open work, in priority order

1. **Fix the residual** - normalise after each addition, then re-run the comparison against 5.53%.
2. **Move the comparison to the GPU** - our normalisation still calls NumPy, which is why that run was CPU-bound at 3.4 ms GPU time inside a 140 ms step.
3. **Settle `reshape`** - step-size sweep to separate a check artefact from a real bug, then build real multi-head attention.
4. **Measured readability** - print audited concept coverage percentage per run.
5. **Safety as training objective** - refusals learned during training with a sealed log, not a post-filter.
