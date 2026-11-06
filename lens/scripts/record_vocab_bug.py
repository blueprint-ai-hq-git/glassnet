"""record_vocab_bug.py — the checkpoint vocabulary bug: found, proven, fixed, migrated.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — A REAL BUG FOUND: CHECKPOINTS STORED VOCAB SIZES BUT NOT VOCAB WORDS

### HOW IT SURFACED

Rebuilding the virtual camera with the BEST VL model (`vqa2_small_ep6.npz`, 88.13%) instead of the
weak one produced, in order:

```
1. BrokenPipeError: [Errno 32] Broken pipe     -> my fault: --out without ".mp4", so ffmpeg could not
                                                  infer the container and died at write time
2. IndexError: list index out of range         -> the real bug
```

### THE REAL BUG

`virtual_camera.py` rebuilt the answer vocabulary from Fashion-MNIST data -- **12 words**. The vqa2
model has **14 answer classes**:

```
cam_base_ep6.npz    answer.W (17, 12)   V=12   [<pad>, <unk>, ankle_boot, ... trouser]
vqa2_small_ep6.npz  answer.W (17, 14)   V=14   [<pad>, <unk>, ankle_boot, ... trouser, yes, no]
                                                                                  ^^^^^^^^^^^
```

**The checkpoint recorded only the SIZE (`V`, `QV`), never the words.** So a loader was forced to
rebuild the words from whatever data it had, and nothing checked that the rebuild agreed.

Two consequences, and the second is worse than the first:

* the sizes disagreed -> `IndexError`, which is merely a crash;
* **if the sizes had happened to agree, the model would have printed the WRONG CLASS NAME with no
  error at all.** A silently mislabelled answer is far more dangerous than a crash in a project whose
  entire purpose is that nothing is hidden.

### WHY THIS WAS INCONSISTENT WITH THE REST OF THE DESIGN

The CONCEPTS already had the correct treatment: stored by NAME and FORMULA, rebuilt from a real spec,
and a missing name is a **hard error** rather than a quiet substitution --
`"refusing to substitute a different concept"`. The vocabularies had none of that. The fix makes them
consistent: the words travel with the weights.

### THE FIX

In `lens/vision.py`:
```
save_vision(net, path, kind, avocab=..., qvocab=...)   # stores the words, and REFUSES to write if
                                                       # the word count disagrees with the weights
check_vocab_match(meta, avocab, qvocab)                # raises on ANY disagreement, and always
                                                       # returns the CHECKPOINT's words to use
```
`load_vision` attaches `net.avocab` / `net.qvocab` so a caller cannot substitute its own.

The check fired correctly on the first attempt after the fix:
```
ValueError: answer vocabulary mismatch: the checkpoint was trained on
['<pad>', '<unk>', ..., 'trouser', 'yes', 'no'], the caller supplied
['<pad>', '<unk>', ..., 'trouser']. Refusing to run -- using either one would map indices to the
wrong words. Use the checkpoint's vocabulary.
```

### THE MIGRATION

Old checkpoints predate the fix. `migrate_ckpt_vocab.py` re-saves each with the vocabulary its OWN
training run used, verified before writing:
```
cam_base_ep6.npz    -> cam_base_ep6_vocabfix.npz      read back: avocab match=True  weights identical=True
vqa2_small_ep6.npz  -> vqa2_small_ep6_vocabfix.npz    read back: avocab match=True  weights identical=True
```
Nothing was overwritten; the originals remain. The size is checked against the weights BEFORE
writing, so a wrong pairing is refused rather than baked in.

### ALSO FIXED: THE CAMERA HAD A TASK MISMATCH

`virtual_camera.py` gained `--task vqa1|vqa2|vqa3`, because each VL task has its own question set AND
its own answer vocabulary. Feeding a vqa2 model vqa1 data was the mismatch above. The camera now
loads the data preparation that matches the model, and compares ANSWER IDS rather than strings --
string comparison is what hid the mismatch until it crashed.

### THE CAMERA, REBUILT ON THE BEST MODEL

```
                              frames   correct            mean confidence
first (weak 69.47% model)         60   41/60  (68.3%)          60.7%
this  (best 88.13% model)        120  102/120 (85.0%)          75.7%
```

It now answers `yes`/`no` questions as well as naming objects, and on each frame it draws the model's
answer, its confidence, and the named concepts that fired.

### THE PATTERN, AGAIN

This is the fourth time a defect has been found by RUNNING the thing rather than by reading it --
and like the others it was in my code, not in the model. `--out` without `.mp4` and a vocabulary
rebuilt instead of stored are both failures of bookkeeping that no amount of model quality fixes.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("CHECKPOINT VOCABULARY BUG — closed")
print("=" * 100)
print("  cause   : checkpoints stored V/QV sizes, never the words")
print("  effect1 : IndexError (loud)")
print("  effect2 : wrong class name with NO error if sizes happened to match (silent, worse)")
print("  fix     : words stored in the checkpoint; check_vocab_match raises on any disagreement")
print("  migrate : 2 checkpoints re-saved, weights verified identical, originals untouched")
print("  camera  : weak 68.3% -> best 85.0%, 120 frames, yes/no + naming")
print("=" * 100)
