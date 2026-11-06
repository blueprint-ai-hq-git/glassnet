"""record_text_concepts.py — the 4-concept text side is ACCEPTABLE (not a bottleneck), and two bugs in
my own audit were caught and fixed on the way.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "vl", "text_concept_audit.json")))
v0 = d["checkpoints"]["mlow_lc0.15"]
v1 = d["checkpoints"]["mlow_lc0.00"]
w = v0["text_rows"]

LOG = f"""
---

## 2026-10-09 — THE 4-CONCEPT TEXT SIDE IS ACCEPTABLE (not a bottleneck). Two bugs in my own audit
### were caught and fixed first.

### THE QUESTION
The "fused" layer has 31 image concepts but only 4 text concepts (asks_what/name/cancer/normal). The
brief's Phase 2 asks for measurable concepts on BOTH sides, so a 4-concept text side deserved
questioning rather than silent acceptance.

### THE PREDICTION I WROTE BEFORE RUNNING
4 text concepts are probably sufficient, because the question templates differ by exactly one
distinguishing word each, so 4 binary cells already encode "which question is this". Adding more
should not improve accuracy. **That prediction held.**

### RESULT (with the metric bug fixed)
```
  lam_concept 0.15                       lam_concept 0.00
  text concepts own_corr                  text concepts own_corr
    asks_cancer   0.795  (strong)           asks_cancer   0.289  (weak)
    asks_normal   0.667  (strong)           asks_normal  -0.106  (decorative)
    asks_what     0.308  (weak)             asks_what    -0.275  (decorative)
    asks_name    -0.293  (decorative)       asks_name    -0.569  (decorative)

  share of |W| carried by text: 12.2%      11.0%
  accuracy full             : {v0['acc_full']*100:.2f}%      {v1['acc_full']*100:.2f}%
  accuracy text side zeroed  : {v0['acc_text_ablated']*100:.2f}%      {v1['acc_text_ablated']*100:.2f}%
  ablation cost             : {(v0['acc_full']-v0['acc_text_ablated'])*100:.2f} pts       {(v1['acc_full']-v1['acc_text_ablated'])*100:.2f} pts

  t2  text side < 5% of |W|      -> FAIL (it is 12.2%)
  t3  ablating text costs > 10 pts -> FAIL (it costs 1.97)
  VERDICT: ACCEPTABLE — a modest contributor, not a bottleneck.
```
The 4 text concepts are mutually independent (off-diagonal corr -0.33, the expected value for four
near-one-hot cells), so they do encode distinct questions. At lam_concept 0.15 two of them genuinely
track their named question (0.795, 0.667) while two are weak (0.308, -0.293).

### TWO BUGS IN MY OWN AUDIT, CAUGHT AND FIXED
1. **Wrong label space for accuracy.** I compared the model's logits against `yte` (the 9-class tissue
   label) and got **4.00%** on a model that measures 66%. `4.00%` is exactly 1/25 — chance level, which
   is what tipped me off. The model's logits are over the ANSWER VOCABULARY (yes/no + class words), so
   accuracy must be scored against `A` — the same metric `train_vl.answer_acc` uses. After the fix the
   audit reports 66.38% / 66.72%, matching the training logs.
2. **Verdict keyed on the wrong gate.** With the metric fixed, t2 FAILED (12.2% > 5%) and my code then
   fell through to "THE TEXT SIDE IS LOAD-BEARING". But t3 also FAILED (1.97 pts < 10), so the text
   side is NOT load-bearing. **A weight share alone does not make a side load-bearing; the ablation
   cost does.** The verdict now keys on t3 only. This is the fourth time today a verdict followed a
   number instead of the gate that was meant to answer the question.

### WHAT THIS ADDS TO THE E18 PICTURE
E18 was about whether the concept layer is as readable as its audit implied. This extends it to the
text side: **the 4 text concepts are independent and mostly honest at lam_concept 0.15, and removing
them costs under 2 points.** The text side is thin but not a limitation worth fixing for this task.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended text-concept audit")
print(f"  acceptable: 12.2% of |W|, ablation cost {(v0['acc_full']-v0['acc_text_ablated'])*100:.2f} pts")
print("  bugs fixed: accuracy label space (yte -> A), verdict keyed on ablation not weight share")
