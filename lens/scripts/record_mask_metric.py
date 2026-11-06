"""record_mask_metric.py — the assistant-only run looked like divergence (ppl 6425 rising) but the
METRIC was wrong. Split by speaker, the mask is clearly working.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "chat", "assistant_ppl_diag.json")))
a, b = d["v12k"], d["asonly"]

LOG = f"""
---

## 2026-10-09 — THE MASK WORKS; THE METRIC WAS LYING. (val ppl is the wrong measure here)

### THE SYMPTOM THAT LOOKED LIKE FAILURE
The assistant-only run showed val ppl **6,425 -> 7,480 and RISING** by epoch 5, while v12k finished
at 98.00. On the standard metric that is divergence, and the tempting move is to kill the run.

### WHY THAT READING WAS WRONG
Validation perplexity is computed over ALL tokens — including the HUMAN's text. The mask deliberately
stops training on human text. So the run was being graded on exactly what it was told to ignore, and
the score had to get worse by construction. The metric cannot detect whether the mask helped.

### THE TEST (split validation loss by speaker, from artifacts, no retraining)
```
  model                ALL tokens    ASSISTANT (judged on)    HUMAN (ignored)
  v12k (24 epochs)       100.32            {a['ppl_asst']:>9.2f}            {a['ppl_human']:>9.2f}
  asonly (8 epochs)     7566.80            {b['ppl_asst']:>9.2f}          {b['ppl_human']:>9.2f}
```

**On the assistant's own turn — the thing a chatbot is actually judged on — the masked model at
epoch 8 is already nearly 2x better than the unmasked model at epoch 24** ({a['ppl_asst']:.2f} ->
{b['ppl_asst']:.2f}). It is dramatically worse at re-predicting the human's words ({a['ppl_human']:.2f}
-> {b['ppl_human']:.2f}), which is precisely the trade the mask asks for.

### THE HONEST CONCLUSION
The mask is WORKING. But two things must be stated plainly:
1. **This comparison is not yet fair on epochs**: the masked run is at epoch 8 of 24, the unmasked
   finished 24. The masked number will move further. The final gate uses the FINISHED run and the
   same n=150-style prompt evaluation as every other chat claim.
2. **"Assistant ppl" is a new metric I introduced here.** It is the right metric for this model, but
   introducing a metric mid-experiment is exactly how a result gets flattered. It is therefore
   recorded as a DIAGNOSTIC, not as a gate pass, and the pre-registered gates (a1 unk rate, a2
   bits/token on shared prompts) still decide the outcome.

### RULE ADDED
A metric computed over tokens the model was deliberately NOT trained on cannot measure that model.
Before comparing two training objectives, confirm the metric scores what each objective optimises.
This is the same family as the verdict-check bugs: a real number, the wrong subject.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended mask/metric record")
print(f"  assistant ppl {a['ppl_asst']:.2f} -> {b['ppl_asst']:.2f} (BETTER, mask working)")
print(f"  human ppl {a['ppl_human']:.2f} -> {b['ppl_human']:.2f} (worse, as designed)")
