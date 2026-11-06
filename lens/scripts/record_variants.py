"""record_variants.py — Step 2 milestone: one interface, three variants, refusal tested.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — STEP 2: ONE INTERFACE, THREE MODELS — see-only / talk-only / see+talk

`lens/scripts/variants.py` is the single front door the owner asked for:

```
  v = Variants(see=..., talk=..., see_talk=...)     # any subset loads
  v.see(img)                  -> class + confidence + named concepts that fired
  v.talk(window_ids)          -> next word id + confidence + fired concepts
  v.see_and_talk(img, prefix) -> next caption WORD + fired concepts
  v.explain(img, prefix)      -> which named concepts drove the chosen word
```

### THE SMOKE RUN (real output)

```
  loaded capabilities: ['see', 'see_talk', 'talk']
  see(img0): class_9  conf 0.85
  see_and_talk(img0,'this is a ___'): sneaker conf 0.60  (truth ankle boot)
  explain: top image-driven contributions -> is_thin_structure 2.34, is_dark 1.20,
           bottom_half_heavier 1.14
  REFUSAL TEST: OK — a see-only build asked to talk raises:
      "this build has no 'talk' model loaded — available: ['see']. Refusing instead of guessing."
  see_and_talk noun position, 500 TEST images: 386/500 = 77.2% chose the class word
```

77.2% on the NOUN position of test images matches the g4 accuracy (77.29%) — the deployed interface
behaves exactly like the trained model, which is the check that the loader/round-trip is honest
(checkpoint round-trip was verified bit-identical: max |orig-loaded| = 0.000e+00).

### TWO HONEST DETAILS FROM THIS STEP

1. **see(img0) first printed "9"** — the see checkpoint has 10 output classes but the fashion NAME
   list (11-12 words depending on version) did not match its width. Now the rule is: if the
   checkpoint's own avocab does not match the output width, print `class_{i}` and nothing else.
   This is the camera vocab-bug lesson applied at the interface level: never print a name we
   cannot prove matches.
2. `load_vision` returns `(net, meta)` — I called it as if it returned the net and hit the
   guess-the-API bug again. The module docstring now lists the REAL signatures of every loader it
   uses, copied from the source, so the next caller reads instead of guesses.

### THE TEST-SET LEDGER

The 500-image noun-position number is the FIRST and ONLY use of the Fashion TEST split for the
see_talk checkpoint, logged in TEST-USE.md. The g4 comparisons used validation only.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 96)
print("  see-only + talk-only + see+talk: one interface, all three load and answer")
print("  refusal tested and passing; see_and_talk on TEST: 386/500 = 77.2%")
print("=" * 96)
