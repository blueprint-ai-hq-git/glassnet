# The copy decay, read off the saved checkpoints (2026-11-13)

The project's record says copy ability peaked at **70.2%** held-out at step 1000 and fell to **46.5%** by step 4000 while
language improved, and that no checkpoint had both. Those numbers came from training logs. Before building a fix, this measures
the same thing from the **checkpoints themselves**, because a fix has to be compared against a reproducible curve, not a printed
line.

## Measured from the checkpoints
`tools/copy_decay_curve.py` rebuilds the copy task's evaluation exactly as the trainer defines it - same vocabulary file, same
taught/held-out word split (lowercase 4-10 letter words, shuffled with rng(0), half never trained on), same three prompt forms,
same item seeds, 600 items per side - and scores each saved checkpoint.

```
checkpoint               taught words   held-out words
early_s0_ep1000.npz         83.0%           71.0%
early_s0_ep2000.npz         83.8%           64.3%
early_s0_ep3000.npz         82.5%           57.2%
early_s0_ep4000.npz         84.3%           45.8%
early_s0_best.npz           83.0%           71.0%     (same weights as ep1000 - it was the best)
early_s0_final.npz          84.3%           45.8%
```

## What this confirms, and what it does not
- **The decay is real and it is in the weights.** Held-out copy falls **71.0 -> 45.8** across three thousand steps (a **25.2
  point** loss) while taught-word copy stays flat at 82.5-84.3%. This is not a logging artefact and not a change of test set.
- **The recorded figures are reproducible to within a point**: I measure **71.0%** at step 1000 against the record's **70.2%**, and
  **45.8%** at step 4000 against **46.5%**. A ~0.8 point offset means my evaluation items are not byte-identical to the trainer's
  (my rebuild reconstructs them from the same seeds and prompt templates rather than reusing the trainer's in-memory arrays). The
  shape and the endpoints agree, so the phenomenon stands - but I am not claiming an exact re-run, and the small gap is stated
  instead of rounded away.
- **What it does not tell us:** which parameters carry the facts. That is the next measurement, and the fix depends on it: if the
  facts live in weights the language task also updates, only a separated store (measured at 96.1 -> 0.0 points of forgetting in
  the controlled experiment, `docs/MEM-SPLIT.md`) can hold them.

## Why this is the right reference for the fix
The controlled split experiment used a 128-fact task with a 24-wide hidden layer. This is the real thing: a 15.5M-parameter model,
4,852 taught words, 4,853 held-out words, and a 25.2-point decay to beat. A separated store that keeps held-out copy at ~71%
across the same 4,000 steps while taught-word accuracy holds at ~83% would be a real result; anything less is a smaller version
of the same failure.

## Reproduction
`.venv-gpu/bin/python3 tools/copy_decay_curve.py` (defaults to the `early_s0_ep*` checkpoints, 600 items per side, GPU); output in
`runs/copy_decay/curve.json`. Evaluation batches of 200 match the trainer's own `accuracy()`.
