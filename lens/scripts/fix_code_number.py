"""fix_code_number.py — CORRECTION: the code model's authoritative number is 4/25, not 8/25.

I quoted 8/25 from `code1_reexec.json`, an artifact written in a PREVIOUS session. Re-running the
authoritative harness (`reexec_code.py`) myself just now gives 4/25 syntax and 4/25 run. Citing a
stale artifact after being burned twice today by stale artifacts is exactly the error this project
keeps logging; it gets fixed the moment it is seen.

AND the re-run surfaced the real blocker, visible in the prompts themselves.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — CORRECTION: code is 4/25, not 8/25 (stale artifact), and the REAL blocker is visible

**What I said:** "code1: 8/25 syntax, 7/25 run."
**What the authoritative harness says when I run it MYSELF, now:** `ALL prompts syntax 4/25, runs
4/25`. The 8/25 came from `code1_reexec.json`, an artifact from a previous session. I re-ran
`reexec_code.py` directly instead of trusting the file. **4/25 is the honest current number.**

### THE REAL BLOCKER IS NOW VISIBLE IN THE PROMPTS
Look at what the model is actually handed (straight from the run):

```
  PROMPT ...'e function should return ` 29 ` . ` ` ` python def <unk> ( number ) : '
  PROMPT ...'gnature : ` def <unk> ( file_path , dataset_name ) - > np . ndarray : '
  PROMPT ...'unk> inputs . write a function : ` ` ` python def <unk> ( a1 , a0 ) : '
```

**23 of the 25 prompts contain `<unk>` in the function signature** — the function NAME has been
deleted by the 6,000-word vocabulary before the model ever sees it. The model is then asked to
continue a signature whose identifiers do not exist, and continuing with `<unk>` is the locally
correct behaviour. Measured on the 2 prompts that are clean, it scores **1/2 (50%)**.

So the 4/25 has two causes, and neither is "the model is stupid":
  1. **Vocabulary too small (6,000 words)** — code identifiers get dropped as `<unk>`, including the
     function names the model is asked to reproduce.
  2. **Training text was half prose** — measured earlier: '`' 6.08%, '<unk>' 5.74%, 'the' 5.55% of
     the corpus. English words crowd out identifier slots.

### THE FIX THAT IS ALREADY PREPARED ON DISK (no download needed)
`data/code/prepared/code_ids_v32000_code.npy` — a **32,000-word, code-only** corpus, already built
(61.6 MB). It addresses BOTH causes at once: 5.3x more vocabulary slots for identifiers, and no
problem-statement prose in the training text. It was prepared in an earlier session and never
trained on. Training it is the next action, and it needs no network at all.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code-number correction (4/25, not 8/25)")
