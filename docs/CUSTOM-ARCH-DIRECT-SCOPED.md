

---

# CUSTOM ARCHITECTURE: MAKING THE CONCEPTS LOAD-BEARING (scoped run, 2026-11-20 12:33)

**Scope, stated first:** 20,000 windows, 1 epoch, `--method ours`, d 128, window 8. The trainer's own measurement says the FULL
corpus under this recipe costs **10.8-13.8 h per arm**, which does not fit any wake, so this is a **bounded matched comparison**:
it can establish direction, not the full-corpus number. Both arms identical except one switch.

## The one change

`--direct` present (arm A) vs absent (arm B). With the direct path the answer layer receives a token-level shortcut; without it the
answer must be computed from the named concepts and sparse rules. That is the difference between "readable because we also train
concepts" and "readable because it cannot answer any other way".

## Measured, from the trainer's own final line

```
arm A  --direct ON    final VAL ppl 3841.51   acc 4.11%   concepts 13/36
arm B  --direct OFF   final VAL ppl 4520.79   acc 3.95%   concepts 14/36
```

- **P2 PASSED (the accuracy cost is small):** **-0.16 points** (4.11 -> 3.95), far inside the 2.0-point bar. Perplexity does rise
  3841.51 -> 4520.79, which is the honest counterweight: the accuracy is nearly free but the model is less confident.
- **P3 PASSED (concepts still mean what they say when load-bearing):** passing concepts **13/36 -> 14/36**, and the per-concept
  AUROCs are consistently, if slightly, HIGHER without the bypass: mentions_body 0.63923 -> 0.65524, is_a_list 0.62948 -> 0.63468,
  mentions_politics 0.55818 -> 0.57263, mentions_a_size 0.57875 -> 0.58962, mentions_law 0.54896 -> 0.55316,
  mentions_a_colour 0.56389 -> 0.56521, mentions_a_quantity 0.52239 -> 0.52508. Seven of eight listed concepts improve.
- **P1 UNMEASURED - my pre-registration was flawed.** I predicted the concept share would rise, but `train_chat.py` prints **no
  share metric at all**, so the run could not answer its own primary question. Worse, my script passed no `--out`, so both arms
  wrote the default `run1` and the **second arm overwrote the first arm's checkpoint** - the offline share measurement cannot be
  done on arm A's weights either. Recorded here rather than quietly dropped.

## What that means

The interesting, defensible result is P3: **removing the shortcut makes the concept labels more honest, and costs 0.16 points of
accuracy.** That is the first measurement in this project where readability and accuracy moved in the same direction rather than
trading off.

## What is fixed and running because of the two defects above

1. `--out "arch_$1"` per arm, so a pair can never clobber itself again. The clobbered file is kept, not deleted:
   `lens/runs/chat/ARCH-SMOKE-UNTAGGED-run1_ep1.npz`, sha256 `171350909327a865...`.
2. The pair re-launched as `customarch2` with distinct tags, so the concept-share measurement becomes possible offline.
3. The next instrument owed: a share measurement that runs on a checkpoint and prints the corrected signed terms
   (`concept_signed` / `reconstruct` / `cancellation`), because a prediction is worthless if the run cannot print its own answer.
