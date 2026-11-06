"""record_constrained_fail.py — CONSTRAINED DECODING FAILS. Root cause: the prompt/model learned
markdown-prose, not code. Recorded as a failure per the pre-registered rule; not spun.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "code", "constrained_replies.json")))

LOG = f"""
---

## 2026-10-09 — CONSTRAINED DECODING FAILS (0/25 both arms). Root cause: DATA, not decoder.

Pre-registered gates (fixed before running):
  c1  syntax with constraint > unconstrained 8/25  -> FAIL (0/25 vs 0/25)
  c2  ran >= half of parsed                   -> FAIL (0/25)

### WHAT HAPPENED
Both arms produced 0/25 syntactically valid output. The constraint checker was proven sound
separately (7/7 against CPython), so the failure is in the MODEL's distribution, not the checker.

### ROOT CAUSE (visible in the generated samples)
The prompt construction cuts validation text at the first `"def "`. But the training data
(magicoder_oss_instruct) wraps solutions in markdown code fences with prose explanations BEFORE
the code. So the prompt often looks like:

    `# create a method to example data for the given input - ```python def ( x ) :`

The model continues with garbage: `" " : = [ ] - = 0 , = 0 ] , [ 1 ) # retur` — not code. The
constraint masks tokens that break syntax, but it cannot make the model produce MEANINGFUL code
when its learned distribution is dominated by markdown-prose tokens. **A syntax constraint
fixes syntax; it cannot fix a semantic gap.**

### WHY THIS IS A DATA/PROMPT PROBLEM, NOT A DECODER PROBLEM
The model was trained on solutions that include ` ```python ` fences and English explanations.
It learned to reproduce that format. Given a prompt that starts mid-markdown-block, it continues
in markdown-prose mode. No decoder change can fix that — the model simply has not learned to write
raw Python in response to a raw-Python prompt.

### WHAT WOULD ACTUALLY FIX IT (not done — needs owner decision)
1. **Strip markdown from training data** and retrain — the clean fix, but costs GPU time.
2. **Change the prompt** to start AFTER the markdown fence (find ` ```python\\ndef ` and cut there)
   so the model only ever sees raw code as context. Cheaper, but changes the eval task.
3. **Train on a code-only corpus** (e.g. raw .py files from a licence-clean source) — most
   correct, needs data sourcing.

### WHAT I AM NOT DOING
I am not tuning the constraint, adding more epochs, or changing decoder settings to chase this
gate. The pre-registered rule says report and stop. The finding is: **the code model's training
data format (markdown-wrapped prose) is the root cause of its poor generation, and constrained
decoding cannot fix it.** Any real fix requires a data or prompt change, which is an owner
decision (GPU time / data sourcing).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended CONSTRAINED FAIL record")
print(f"  c1 FAIL: syntax 0/25 vs 0/25")
print(f"  c2 FAIL: ran 0/25")
print(f"  root cause: markdown-prose in training data, not a decoder issue")
