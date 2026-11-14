# The 36 concept activations carry almost no generalisable next-token information

Date: 2026-11-14. Tools `tools/readout_fix_probe.py` (post-hoc repairs) and `tools/concept_probe.py` (linear probe), runners in
`lens/scripts/`, predictions `lens/runs/READOUT-FIX-PREDICTIONS.txt` and `lens/runs/CONCEPT-PROBE-PREDICTIONS.txt`, outputs
`lens/runs/READOUT-FIX-PROBE.txt`, `lens/runs/CONCEPT-PROBE.txt`.

## Test 1 - can the model's own readout be repaired post-hoc? No.

Four repairs applied to the same concept logits `h @ W + b`, d512 (2,000 held-out positions):

```
baseline (argmax as-is)          5.25%
no_unk (exclude <unk>)           3.15%
centred (remove column means)    0.60%   (d1024: 0.15%)
no_unk + centred                 0.65%   (d1024: 0.30%)
zrow (per-row standardise)       5.25%
```

**Every repair fails, three of them make it worse.** All three predictions (P1 >8%, P2 >12%, P3 <half of full) failed, and the
falsifier fired: every repair stayed at or below 6.0%. Being below the `<unk>` base rate after removing `<unk>` is the telling
number - there is no hidden ranking underneath that column.

## Test 2 - but is the information in `h` at all? Almost none.

A failed readout and an uninformative activation look identical through test 1, so the activation vector was probed directly: my own
softmax regression (plain SGD/Adam, written in the tool) from the 36 concept activations to the next token, fitted on 20,000 TRAIN
positions and evaluated on the same 2,000 held-out positions.

**First version of this probe was itself broken and was thrown away**: 400 steps scored 2.25% while predicting 185 distinct tokens,
which is an underfitted probe, not a finding. The gradient also had a shape bug (`index 37 is out of bounds for axis 0 with size 37`)
because the (37, V) weight gradient was indexed as if per-sample. Both fixed before any number was trusted.

Fitted to convergence (3,000 steps):

```
probe TRAIN accuracy        18.32%     <- the probe CAN fit, so it is not broken
probe HELD-OUT accuracy      4.45%     <- below the <unk> base rate
on the 200 commonest tokens  6.54%     (constant most-common-token baseline 5.25%)
distinct tokens predicted      224
```

- **P4 FAILED**: I predicted at least 15.0% held-out if the information were present. It is **4.45%**.
- **P5 HELD**: the activations carry almost nothing for next-token prediction.

**Train 18.32% against held-out 4.45% means the probe memorises its 20,000 rows and generalises to nothing.** The 36 activations
are not a compressed representation of the next token; they are close to a dead feature vector for prediction purposes.

## What this settles

The earlier document established that the *readout* was degenerate. This one establishes the harder fact: **the concept layer
itself carries almost no generalisable predictive information, so no readout fix can rescue it.** Combined with the coverage result
(direct pathway 82-91%), the chat model's readable layer is currently decorative for prediction:

```
direct pathway share of the decision     82-91%        (CHAT-EXPLAIN-CAUSAL.md)
concept path answer                     constant <unk> (CONCEPT-READOUT-DEGENERATE.md)
concept activations, held-out probe      4.45%         (this document)
full model accuracy                     28.25%         (DIRECT-PATH-COST.md)
```

## Consequences, in dependency order

1. **Do not spend a GPU hour on a readout tweak.** It cannot help; the information is not there to read.
2. The next readable-chat experiment must change **what the concepts are forced to carry**, not how they are read. The natural
   design is a **bottleneck**: the direct pathway must be constrained *and* the concept vector must be the only route to the answer,
   so gradients cannot route around it - which is exactly the image model's structure (concepts -> rules -> classes) that measured
   81.95% +/- 0.53 at 10,166 parameters.
3. Until such an arm trains, every public statement about chat readability must say **the concepts do not yet carry the answer**.

## Caveats

- One checkpoint (d512, seed 0), 20,000 train and 2,000 held-out positions, 36 features.
- The probe is **linear**. A non-linear probe could find information this one cannot; nothing here rules that out.
- "Almost no information" is measured against next-token prediction on this corpus, which is the model's actual task.
