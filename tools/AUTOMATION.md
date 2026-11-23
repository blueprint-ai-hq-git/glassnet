# Automation is not published here

The queue driver, the chain runner, the notifier and the freeze watchdog are **operator-side tooling**. They contain machine-specific
details (local paths, GPU vendor commands, notification routing) and they are not part of the model or the experiments, so they are not
shipped in this repository.

What IS published is every script that produces a number in the documentation:

| script | what it measures |
|---|---|
| `lens/explain_emit.py` | the written-reason ladder: when a model writes its reason as tokens, and whether it is honest |
| `lens/concept_honest.py` | does a concept neuron mean its own name, with planted liar neurons as the instrument check |
| `lens/concept_prune.py`, `lens/concept_prune_cost.py` | how few concepts describe the state, and what pruning costs in accuracy |
| `lens/concept_name_match.py` | match stored concept columns to their named definitions by measurement, not assumption |
| `lens/width_trained.py` | does widening the readable layer help when the answer head reads ONLY the concepts |
| `lens/verify_library.py` | the 5-20 line library and `explain()`, end to end, including save/load and export |
| `tools/direct_ablation.py` | does the readable route alone reproduce the model's own answer |
| `tools/concept_holds_answer.py` | does a best-possible readout from the concepts recover the answer |
| `tools/concept_readout_nonlinear.py` | the same without any linearity assumption |
| `tools/hidden_holds_answer.py` | how wide a probe of the concept layer's input has to be |
| `tools/concept_rank.py` | per-concept attribution of the decision margin, with the accounting check that must close |

Reproduce any published number with the script named beside it. Sample sizes, seeds and the pre-registered bars are printed by each script
before it reports a verdict.
