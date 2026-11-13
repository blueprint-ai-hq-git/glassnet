# Where the parameter budget actually goes: half the model does nothing (2026-11-13)

The target is a model under 50M parameters that is actually good, so the question that decides the design is: **which parameters earn
their place?** This measures it for the language objective the same way the copy line already did for copy - each tensor replaced by
its own mean (scale kept, variation destroyed), then language accuracy on the held-out matched split re-measured.

## Language ablation - checkpoint `early_s0_ep1000`, baseline 16.53%, 16,354,001 parameters
```
tensor        params      language after   damage     share of the model
emb           6,145,024       0.00%        +16.53          37.6%
ptrW            262,144       2.13%        +14.40           1.6%
Wq              262,144       5.67%        +10.87           1.6%
Wk              262,144       6.07%        +10.47           1.6%
concept.W       589,824      16.13%         +0.40           3.6%
answer.b         12,002      16.33%         +0.20           0.1%
concept.b            36      16.60%         -0.07           0.0%
answer.W        432,072      16.60%         -0.07           2.6%
Wg / bg / kamp        1      16.53%         +0.00           0.0%
dmix          8,388,608      16.53%         +0.00          51.3%
```

## The two headline facts
1. **`dmix` is 51.3% of the model and is worth 0.00 points of language accuracy.** It was already measured as 0.0 points of copy
   damage with 0.0000 drift (`docs/COPY-WEIGHTS.md`). It is now inert on **both** objectives. Half the parameters are carrying nothing
   measurable.
2. **The pointer matrices are 4.8% of the parameters and carry 10.5-14.4 points of language accuracy.** Per parameter, `ptrW`/`Wq`/
   `Wk` are worth roughly **100x** what `dmix` is worth. They are also the tensors language training overwrites most (drift 0.57 for
   `ptrW`), which is why the copy line had to freeze them.

## Cross-referenced with the copy ablation (`docs/COPY-WEIGHTS.md`, same checkpoint)
```
tensor        share    copy damage   language damage   reading
emb           37.6%       +69.5           +16.53        both objectives depend on it
ptrW/Wq/Wk     4.8%       +69.5       +10.5 to +14.4   both objectives; best value per parameter
concept.W      3.6%       +15.0            +0.40        mostly the copy/generalising path
answer.W       2.6%        +0.8            -0.07        effectively unused by both
dmix          51.3%        +0.0            +0.00        dead weight on both
```

## What this means for the 50M design
- **Do not scale `dmix`.** A 50M model built the current way would spend about **26M parameters on a component measured at zero** on
  both objectives.
- **Spend the budget on width and on the pointer path.** Width multiplies `emb` (the only tensor that kills both objectives when
  removed) and `ptrW`/`Wq`/`Wk` scale as `d^2`, so they grow with width for a tiny share of the total.
- A concrete next configuration at `d = 1024` **without** `dmix`: `emb` 12.29M + `concept` 1.34M + `answer` 0.49M + `ptrW` 1.05M +
  `Wq`/`Wk` 2.10M = **about 17.3M parameters**, well inside the 50M budget with room left to widen further or deepen the concept
  layer.

## Honest limits
- This is one checkpoint (`early_s0_ep1000`), and mean-ablation is blunt: it destroys a tensor completely, so "0.00 damage" means the
  tensor's variation is not needed by this model on this task, not that the component is useless in every design.
- `answer.W` and `concept.b` show **negative** damage (-0.07), i.e. ablating them slightly *helps*. That is within measurement noise
  on 1,500 windows and should be read as "no measurable contribution", not as a win.
- `dmix`'s inertness is measured on this corpus and this training setup. It was built as a direct pathway; it may simply never have
  been switched on in a way this objective could use. Either way, it has not earned 8.4M parameters here.

## Reproduction
`tools/lang_ablation.py --windows 1500` (GPU, ~4 s), output `runs/lang_ablation/lang_early_s0_ep1000.json`. Copy numbers from
`tools/copy_ablation.py`, documented in `docs/COPY-WEIGHTS.md`.
