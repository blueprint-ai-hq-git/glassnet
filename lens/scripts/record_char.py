"""record_char.py — character level: the architecture argument is strong, the quality result is not in yet.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 18:40 — OPTIMISATION 2: CHARACTER LEVEL. THE ARCHITECTURE ARGUMENT HOLDS; QUALITY IS UNPROVEN.

The owner said "optimise it, optimisation is key". The first diagnostic found the concept layer
compressing 114:1 where the scaling win used 50:1. The second-biggest number in that diagnostic was
that **93% of the word model is the embedding table**, and the answer layer must pick **1 of 12,000
words through a 9-wide pipe** (13.6 bits).

Character level attacks both at once.

### THE ARCHITECTURE ARGUMENT, MEASURED

```
                            word d=128 K=9     char d=64 W=32 K=35
  vocabulary                       12,000                    432
  embedding share                   92.9%                   ~19%
  concept layer share                0.6%                   ~67%
  bottleneck ratio                  114:1                   29:1
  <unk> rate                        1.92%                   0.00%
  corpus                       13.3M tokens          60.9M characters
```

**At character level the named concept layer stops being 0.6% of the network and becomes the part
that carries the model.** That is the entire point of this architecture, and the word-level
configuration was structurally unable to express it. There is also no `<unk>`, so the 1.92% of
tokens currently discarded is kept.

### THE QUALITY RESULT IS NOT IN, AND THE SMOKE TEST DOES NOT SUPPORT THE IDEA

The character smoke run (1 epoch, 8,000 windows, ~150k params) reached **ppl 43.49**, which looks
far better than the word model's **353.28**. **It is not.** Perplexity is not comparable across
tokenisations -- one predicts 1 of 432 characters, the other 1 of 12,000 words.

In the common currency, **bits per character of text**:
```
                              word-level      char-level
  perplexity                      353.28           43.49
  bits per token                   8.465           5.443
  chars per token                  4.576           1.000
  BITS PER CHARACTER               1.850           5.443     <- lower is better
```

**On the data actually seen, WORD level wins by 3x.** The character model is a smoke test: 1 epoch,
8,000 windows, 28x less data and 11x fewer parameters than the word run.

**So the honest position is: the architecture argument is strong and measured, and the quality
argument is NOT made yet.** A full character run is queued to settle it, with the prediction written
first: **if bits per character does not beat 1.850, character level is the wrong choice for this
corpus and that gets said plainly.**

### WHAT WAS BUILT

- `chat_data_char.py` -- character tokeniser. **432 distinct characters, 60,949,068 characters**
  over 86,372 conversations. Vocab built from TRAIN text only.
- `chat_concepts_char.py` -- concept targets at character level. The same word lists, matched as
  substrings on decoded window text with letter/apostrophe boundaries. **Stated as an approximation
  rather than assumed to be equivalent to tokenisation.**
- `chat_concepts_wide.py` -- the concept set widened from 12 defined (9 kept) to **46 defined**.
  At character level **35 kept, 11 dropped** by the same 0.5% support floor.
- `train_chat_char.py` -- the character trainer, with its own local fidelity audit because
  `lens.audit` derives targets from token ids and character concepts are declared as regexes.
  **Same bars: AUROC 0.55, firing floor 0.02. Not adjusted.**

### BUGS (28-30)

28. `Concept` requires token ids and the character path passed an empty list ->
    `ValueError: concept 'is_a_question' has no token ids`. Fixed with documented placeholder ids
    plus a local audit against the DECLARED targets.
29. `NameError: cannot access free variable 'vmap'` -- the placeholder ids were computed before
    `vmap` was defined. Fixed by moving the definition up.
30. `CharNet.forward` called `self._net.forward(X)` twice, building two graphs per call. Wasteful
    and would have doubled the forward cost; fixed.

### THE QUEUE, IN ORDER

1. distillation seed 2 (running)
2. optimisation grid: d=384 K=48, d=384 K=9, d=128 K=48  -- isolates the concept count
3. full character run, compared in bits per character
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
