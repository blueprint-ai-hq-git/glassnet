"""record_assistant_only.py — the assistant-only masking experiment: built, verified, launched.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — ASSISTANT-ONLY LOSS MASKING: built, verified, launched

### THE PROBLEM (found by reading train_chat.py, not guessed)
The chat model's next-token loss is computed over the ENTIRE window. So the model spends capacity
learning to re-predict the HUMAN's text as well as its own replies. A chatbot is judged on its own
turn, so training on the user's words is wasted work.

### WHAT WAS BUILT
A `--assistant-only` flag that weights the loss to the assistant's own tokens. Three pieces:
1. `softmax_ce()` gained a `sample_weights` parameter — per-POSITION weights, deliberately SEPARATE
   from the existing `class_weights` (which is per-LABEL). Collapsing the two would have silently
   changed the medical runs that depend on label semantics. First attempt used class_weights and
   crashed with `IndexError: index 8141 out of bounds` — the weights were being indexed by label
   instead of by row. That crash is why the two are now distinct arguments.
2. `LensNet.loss()` and the ChatNet wrapper pass the weights through.
3. `train_chat.py` builds the mask by walking the token stream and flipping a speaker flag on
   `assistant :` / `human :` pairs.

### VERIFIED BEFORE TRAINING (a degenerate mask would look like a normal run)
`verify_assistant_mask.py`, on the real corpus:
```
  corpus tokens        13,318,447
  marked assistant      9,549,325  (71.7%)
  marked human/other    3,769,122  (28.3%)

  real conversation, mask following the turns:
    text : human : hi , i want to learn to play <unk> . can you teach me ? assistant : i can , ...
    mask : H H H H H H H H H H H H H H H H H H H A A A A A A A A A A A A A A A A A A A A A A A A A H H ...
  PASS: marks a real subset, not nothing and not everything
```

### STATED UP FRONT, SO IT IS NOT OVERSOLD
Assistant text is **71.7%** of the corpus, so masking human targets away reweights only the
remaining **28.3%**. This is a MODEST change, not a dramatic one. A large effect is not expected;
the honest expectation is a small improvement or none. The gates will decide.

### GATES (pre-registered before the run)
  a1  unk rate on the shared prompts lower than v12k's 3.2%
  a2  bits/token within +0.30 of v12k's 6.61
  a3  if a1 fails -> masking does not improve reply quality; report and STOP (no weight tweaking to
      chase the gate)
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended assistant-only record")
