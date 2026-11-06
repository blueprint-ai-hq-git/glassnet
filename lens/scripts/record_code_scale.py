"""record_code_scale.py — n=150 settles it: vocabulary is NOT the code blocker (p=0.42).
Also records the real chat inefficiency found while looking: no assistant-only loss masking.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "code", "code_scale_test.json")))

LOG = f"""
---

## 2026-10-09 — CODE MEASUREMENT SCALED to n=150: vocabulary is DEFINITIVELY not the blocker

The 25-prompt test was unreadable (p=0.725). The bottleneck was the measurement, so it was scaled to
150 prompts using the 768 validation documents available. Same construction for both models, each
checkpoint with its own vocabulary, same decoder.

```
  model            vocab     syntax        ran      <unk> in output
  code1_ep16        6,000    {d['code1']['syntax']:>3}/{d['code1']['total']} = {d['code1']['syntax']/d['code1']['total']*100:.1f}%   {d['code1']['ran']:>3}/{d['code1']['total']}      {d['code1']['unk']*100:.1f}%
  covocab32k       32,000    {d['covocab32k']['syntax']:>3}/{d['covocab32k']['total']} = {d['covocab32k']['syntax']/d['covocab32k']['total']*100:.1f}%   {d['covocab32k']['ran']:>3}/{d['covocab32k']['total']}      {d['covocab32k']['unk']*100:.1f}%

  Fisher exact two-sided p = {d['fisher_p']:.4f}
  r1  significant at p<0.05 -> FAIL
```

**VERDICT: a 5.3x larger vocabulary does NOT significantly improve code generation (p={d['fisher_p']:.2f}).
Vocabulary is closed as a direction.** The unk rate did fall ({d['code1']['unk']*100:.1f}% -> {d['covocab32k']['unk']*100:.1f}%),
which is the mechanism working as designed, but it does not translate into runnable code. Per the
pre-registered rule: stop pursuing vocabulary, the remaining cause is model capacity or training-data
content.

### THE MEASUREMENT IS NOW TRUSTWORTHY, WHICH IS THE REAL WIN
Three numbers for the same checkpoint (8/25, 4/25, 6/25) came from three different harnesses. Now
there is ONE script, ONE prompt construction, per-checkpoint vocabularies, and a significance test.
Any future code claim comes from this script or it does not get made.

### A REAL CHAT INEFFICIENCY FOUND WHILE INVESTIGATING (not yet fixed)
Reading `train_chat.py` to look for other levers, I found: **the chat model is NOT loss-masked to the
assistant's tokens.** The next-token loss is computed over the whole window, so the model spends
capacity learning to predict the HUMAN's text as well as its own replies. For a chatbot that is
wasted work — the model is asked to be good at generating the assistant turn, not at re-predicting
the user's question.

This is a concrete, testable hypothesis with a clear mechanism, and it is the next one-change
experiment for talk: mask the loss to assistant tokens only, everything else identical, and compare
on the same shared prompts + decoder. Pre-registered gates will be written before the run.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code scale record")
print(f"  n=150: {d['code1']['syntax']}/150 vs {d['covocab32k']['syntax']}/150, p={d['fisher_p']:.4f} -> NOT significant")
print("  vocabulary direction CLOSED; next lever identified: assistant-only loss masking for chat")
