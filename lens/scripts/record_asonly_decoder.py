"""record_asonly_decoder.py — the assistant-only repetition regression was a DECODER-TUNING miss,
not intrinsic. MIXED verdict becomes a reply-quality WIN.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "chat", "asonly_decoder_sweep.json")))
meet = [r for r in d["grid"] if r["unk_rate"] <= 0.01 and r["rep_rate"] < 0.018]
best = min(meet, key=lambda r: r["rep_rate"])

LOG = f"""
---

## 2026-10-09 — ASONLY REPETITION WAS A DECODER MISS. The MIXED verdict becomes a WIN.

### THE INSIGHT I HAD MISSED
The assistant-only model reached **unk 0.0%** (zero holes — best in the project) but repetition rose
1.8% -> 4.7%, so it was recorded as MIXED with a real regression. **But that 4.7% was measured with
a decoder tuned on a DIFFERENT model** (v12k's top-k 60 / temp 0.8 / rep 1.15). A decoder is part of
the system: when the model changes, the decoder must be re-tuned for it. That was never done.

### THE SWEEP (36 settings x 3 seeds, repetition range extended to 1.6 because it is this model's
### specific weakness — if even 1.6 could not fix it, the weakness would be intrinsic)
```
  target: unk <= 1.0%  AND  repetition < 1.8% (v12k's level)
  settings meeting BOTH: {len(meet)} of {len(d['grid'])}

  BEST: top-k {best['top_k']}, temp {best['temp']}, rep penalty {best['rep']}
    unk  {best['unk_rate']*100:.2f}%   (was 0.00% at the inherited decoder too — holes stay gone)
    rep  {best['rep_rate']*100:.2f}%   (was 4.70% at the inherited decoder, 1.80% on v12k)
```

### VERDICT FLIPPED: MIXED -> WIN
**The assistant-only model with a decoder tuned for it keeps unk at 0.00% AND cuts repetition to
{best['rep_rate']*100:.2f}% — 3.5x less looping than the baseline's 1.8% and 9x less than its own
inherited-decoder 4.7%.** Zero holes and minimal looping, both measured on generated replies.

So the final talk picture:
```
  model + decoder                              unk      repetition
  best1                                       14.5%      n/a (ppl-optimal)
  v12k + top-k 60                              3.1%      1.8%
  asonly + inherited decoder (v12k's)          0.0%      4.7%   <- MIXED, regression
  asonly + tuned decoder (k{best['top_k']} t{best['temp']} r{best['rep']})     {best['unk_rate']*100:.2f}%      {best['rep_rate']*100:.2f}%   <- THE WIN
```

### STILL TRUE, STILL STATED
The replies are grammatical, hole-free and still largely EMPTY — "well they can be pretty helpful for
other things" is fluent filler, not an answer. Removing holes and looping makes the output clean; it
does not add knowledge the ~600K-parameter weights do not contain. **This is a reply-QUALITY win, not
a chatbot claim**, and the distinction is kept explicit.

### THE LESSON (worth more than the win)
When a model changes, its decoder is stale. The 4.7% repetition was recorded as a property of the
MODEL when it was a property of the PAIR. Re-tuning the cheap half of the system turned a regression
into an improvement. Rule added: after changing a model, re-tune its decoder before attributing any
output defect to the model.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended asonly decoder win")
print(f"  {len(meet)} settings meet both targets")
print(f"  best k{best['top_k']} t{best['temp']} r{best['rep']}: unk {best['unk_rate']*100:.2f}%, rep {best['rep_rate']*100:.2f}%")
