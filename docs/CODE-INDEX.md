# CODE INDEX — the machinery behind every number published here

Every result document in `docs/` names the file that produced it. This is the index of the code itself, so nothing has to be taken
on trust. All of it is our own implementation: no model library, no autograd framework.

## The architecture and its certificates

| file | what it is |
|---|---|
| `lens/glass_block.py` | our own attention block: LayerNorm, multi-head causal attention, MLP, residuals, and **our own KV cache**. 544 named parameters in the reference config, nothing else to learn. Verified against a hand-computed reference (4.441e-16), cache equivalence (1.192e-07), LayerNorm by hand (0.000e+00). |
| `lens/glass_grad.py` | the backward pass for that block, written longhand. Certified against finite differences: worst relative error **8.815e-07**, input gradient **2.650e-09**. |
| `lens/glass_lm.py` | the trainer: embedding (full or **factored** `E = A @ B`), learned position table, N blocks, tied output, our own momentum SGD, and the optimiser variants that were tested and rejected (per-tensor, per-element, gain rule). |
| `lens/memg.py` | the **memorise-versus-generalise** trap dataset and switch: rules that can only be generalised, facts that can only be stored, and the decision rule between them. |
| `tools/anatomy.py` | inventories a checkpoint tensor by tensor: role, parameter share, state carried per token, and what is opaque. This is what produced the finding that the readable layer is 1.7% of the model. |

## How to run the certificates yourself

```bash
export GLASSNET_ROOT=/path/to/the/project
cd "$GLASSNET_ROOT"
PYTHONPATH=. python3 lens/glass_block.py        # three proofs, prints its own numbers
PYTHONPATH=. python3 lens/glass_grad.py         # finite-difference check of every gradient
PYTHONPATH=. python3 tools/anatomy.py <ckpt.npz>  # tensor inventory
```

Scripts under `lens/scripts/` are the experiment runners that produced the published runs. They expect `GLASSNET_ROOT` to point at the
project and are launched through a job runner so a long run survives the session.
