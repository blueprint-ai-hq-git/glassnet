# Which weights carry copying, and which ones language training rewrites (2026-11-13)

Two measurements on the real copy checkpoints, because the fix has to know what to protect before it protects anything.

**Context that matters:** the decay measured from the checkpoints is in **held-out** words (71.0% -> 45.8%) while **taught** words
stay flat at 82.5-84.3% (`docs/COPY-DECAY-CURVE.md`). Held-out words never appear in any training window, so a memorisation table
cannot help them. What decays is the copy MECHANISM, and a store that protects facts would have fixed a problem that is not the
one being measured.

## 1. Ablation: where copying lives (checkpoint ep1000, baseline taught 83.8% / held-out 69.5%)
Each parameter tensor is replaced by its own mean - scale kept, variation destroyed - and both accuracies re-measured.

```
tensor        params      taught after   held-out after   held-out damage
emb           6,145,024        0.0%           0.0%            +69.5
ptrW            262,144        0.0%           0.0%            +69.5
Wq              262,144        0.0%           0.0%            +69.5
Wk              262,144        0.0%           0.0%            +69.5
concept.W       589,824       84.8%          54.5%            +15.0
answer.W        432,072       83.5%          68.8%             +0.8
concept.b            36       83.8%          69.5%             +0.0
Wg / bg / kamp    1 each      83.8%          69.5%             +0.0
dmix          8,388,608       83.8%          69.5%             +0.0
```

Two things stand out:
- **`concept.W` is worth 15.0 points of HELD-OUT copy while changing taught-word copy by +1.0.** It carries the generalising
  copy behaviour, not the memorised answers. That is exactly the profile of the thing that decays.
- **`dmix` is 8.4M parameters - 51% of the model - and ablating it changes neither accuracy at all.** Half the parameters are
  doing nothing on this task, which is its own finding.

## 2. Drift: what the language phase actually rewrote (ep1000 -> ep4000, relative L2 change)
```
tensor        params      rel drift   mean|w| ep1000   mean|w| ep4000
answer.b        12,002        1.2688        0.01070          0.01195
bg                   1        0.7214        2.08729          3.59302
ptrW           262,144        0.5663        0.01610          0.02756
concept.b           36        0.4325        0.12108          0.14331
concept.W      589,824        0.2691        0.01470          0.01522
Wk             262,144        0.2420        0.00784          0.01291
Wq             262,144        0.2297        0.00797          0.01267
emb          6,145,024        0.0962        0.04869          0.04932
answer.W       432,072        0.0166        0.18794          0.18793
Wg                   1        0.0843        4.10664          4.45263
kamp                 1        0.0000        0.04833          0.04833
dmix         8,388,608        0.0000        0.05048          0.05048
```

## The two measurements agree on the mechanism
The tensors that carry copying (`ptrW`, `Wq`, `Wk`, `concept.W`) are exactly the tensors the language phase moves most
(relative drift 0.23-0.57), while the tensors that carry the memorised answers barely move: `emb` 0.096 and `answer.W`
**0.0166**. That is the whole decay in one line:

- memorised answers live in weights language training barely touches -> taught-word copy stays at ~83%,
- the copy mechanism lives in weights language training rewrites -> held-out copy falls 25 points.

It also explains why the earlier separated-store result (96.1 -> 0.0 points of forgetting, `docs/MEM-SPLIT.md`) is real but does
not apply here: that experiment decayed the FACTS, this task decays the MECHANISM.

## What the fix must therefore do
Freeze or isolate the copy path - `ptrW`, `Wq`, `Wk`, `concept.W` - during language-only training, and measure whether held-out
copy holds near 69-71% while taught accuracy and language stay put. That is a targeted experiment with a clear prediction:
freezing those four should preserve held-out copy; freezing `answer.W`/`dmix` (which barely move and do not carry copy) should
change nothing, and is the control that proves the freeze is doing the work rather than the act of freezing.

## Honest limits of these two measurements
- Mean-ablation is blunt: it destroys a tensor completely, so "0.0% after" on `emb` says copying needs embeddings, not that
  embeddings are the copy circuit.
- Weight drift is not causality. A tensor can move a lot and matter little (`answer.b`, drift 1.27, damage 0.8). The pairing of
  the two measurements is what makes the story credible; neither alone would.
- `dmix` doing nothing here does not mean it does nothing anywhere - only that it has no role in this task and was not updated in
  this interval.

## Reproduction
`tools/copy_ablation.py --items 400` (mean-ablation ranking) and the drift table from the two checkpoints with a small numpy
script; baselines differ slightly from the 600-item curve (69.5% vs 71.0%) because fewer evaluation items are used.
