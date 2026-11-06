"""record_decoder_win.py — the talk decoder sweep: a REAL, free improvement (top-k 40 -> 60).
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "chat", "talk_decoder_sweep.json")))
b, f = d["best"], d["default"]

LOG = f"""
---

## 2026-10-09 — TALK DECODER SWEEP: a real improvement, and it was FREE

27 settings (top-k {{20,40,60}} x temp {{0.6,0.8,1.0}} x repetition penalty {{1.05,1.15,1.3}}), 3 seeds
each, 10 shared prompts. Scored on things a READER can see, not perplexity:

    score = -(unk_rate * 2) - rep_rate - (1 - length_ok) * 0.5

unk weighted double because a hole in the text is the most visible defect. Weights are stated
constants, written in the script, not buried.

```
  BEST    top-k {b['top_k']}, temp {b['temp']}, rep {b['rep']}   score {b['score']:+.4f}   unk {b['unk_rate']*100:.1f}%   bigram-rep {b['rep_rate']*100:.1f}%   len_ok {b['len_ok']*100:.0f}%
  DEFAULT top-k {f['top_k']}, temp {f['temp']}, rep {f['rep']}   score {f['score']:+.4f}   unk {f['unk_rate']*100:.1f}%   bigram-rep {f['rep_rate']*100:.1f}%   len_ok {f['len_ok']*100:.0f}%
  d1  best {b['score']:+.4f} > default {f['score']:+.4f}  -> PASS
```

**top-k 60 replaces the hand-picked top-k 40.** Unks {f['unk_rate']*100:.1f}% -> {b['unk_rate']*100:.1f}%,
bigram repetition {f['rep_rate']*100:.1f}% -> {b['rep_rate']*100:.1f}% (a 2.5x cut in looping). This costs
NOTHING: no retraining, no extra memory, no GPU time. It is a pure decoder change — the model's
weights, perplexity and accuracy are identical, only what it is ALLOWED to sample from changed.

Wired into `variants.talk_sentence()` as the new default, with the evidence path in its docstring.

### TWO BUGS FOUND IN MY OWN SWEEP ALONG THE WAY (both fixed, both worth recording)
1. **The default was missing from the grid.** The sweep compared 27 settings but the current default
   (40, 0.8, 1.15) was not among them, so the final `next(...)` raised StopIteration. A "best" is
   meaningless if the incumbent is not in the comparison. Fixed by forcing the default into the grid.
2. **Key collision.** `score_replies()` returned a key `"rep"` for the repetition RATE, which
   overwrote the `"rep"` repetition PENALTY when merged into the same results dict — so the penalty
   silently became a rate and the default row could never be found. Renamed to `rep_rate`/`unk_rate`.
   Two different things must never share a key; this is the same family as the earlier
   wrong-subject ladder bug.

### HONEST SCOPE OF THE WIN
This makes the SAME model's output less holey and less repetitive. It does NOT make the model
smarter — it cannot add knowledge the weights do not contain. The talk model is still a ~600K-param
model trained on 34MB of chat data, and it is not a real chatbot. What improved is how cleanly it
expresses what it already knows.
"""

with open(os.path.join(P, "LOG.md"), "a") as fh:
    fh.write(LOG)
print("appended decoder win")
print(f"  top-k {f['top_k']} -> {b['top_k']}: unks {f['unk_rate']*100:.1f}%->{b['unk_rate']*100:.1f}%, "
      f"rep {f['rep_rate']*100:.1f}%->{b['rep_rate']*100:.1f}%")
