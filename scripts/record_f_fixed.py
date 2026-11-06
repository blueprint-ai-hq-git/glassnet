"""record_f_fixed.py — record the corrected Target F concept result.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-05 22:25 — TARGET F CONCEPT CLAIM: RE-TESTED PROPERLY. THE CONCEPTS SURVIVE.

After fixing the checkpoint rule (`F_CKPT_RULE=last`: no early stop, all 40 epochs, so the audit
measures a trained model rather than an epoch-1 snapshot) and using the corrected loader, the
readable student was re-trained on the SAME generated corpus and audited:

```
                     R1      R5      coverage
Target E (real)     7/9     6/9     100.0000%
F student (all)     5/9     6/9     100.0000%   <- was reported as 0/9 from a broken audit
```

Per concept (F student, rule=all, seed 0):
```
col concept                  AUROC   contrib   firing     R1    d_all     R5
  0 pronoun_subject         0.5690    0.0000   0.0000   FAIL   0.0000   FAIL
  1 is_a_dialogue_line      0.8310    1.0381   0.0046   FAIL   0.4343   PASS
  2 pronoun_object          0.5624    0.0000   0.0000   FAIL   0.0000   FAIL
  3 is_a_negation           0.5066    0.0000   0.0000   FAIL   0.0000   FAIL
  4 is_archaic_address      1.0000    7.1366   0.0771   PASS   0.9836   PASS
  5 contains_a_wh_word      1.0000    9.8696   0.0522   PASS   0.9978   PASS
  6 mentions_speech         1.0000    9.8880   0.0381   PASS   0.9987   PASS
  7 mentions_giving         1.0000   11.6432   0.0375   PASS   0.9992   PASS
  8 mentions_knowing        1.0000   11.7855   0.0338   PASS   0.9991   PASS
```

**CONCLUSION, CORRECTED:** the concepts **survive a generation of self-training**. R5 is
**identical** to real-data training (6/9 both); R1 is 5/9 against E's 7/9. `is_a_dialogue_line`
passes R5 (0.4343) but misses the R1 firing floor (0.0046 < 0.02).

**The three that fail are the same weak ones seen before:** `pronoun_subject` (0.5690),
`pronoun_object` (0.5624), `is_a_negation` (0.5066) — the same family as the concepts that
broke in Targets D and E. The concepts that hold are the strongest ones.

**WITHDRAWN, definitively:** "all nine concepts died" and every framing built on it. That was
**two bugs in my own evaluation** (checkpoint-key mismatch + best-LM checkpoint rule), not a
property of the model.

**WHAT STILL STANDS — the LM collapse.** The student trained on generated data reaches ~1800 ppl
against the teacher's 81.51 and the real-data control's 76.27. The owner's self-improvement goal
still fails; the reason is the DATA, not the concepts.

**Two fixes made:** `f_train.py` now takes `F_CKPT_RULE=best_lm|last` with the defect documented
in the code, so an audit can no longer silently measure an untrained model.

**Artifacts:** `runs/t3/f_fixed.log`, `ckpt/student_readable_all_f2_seed0.npz`,
`runs/t3/f_audit.json`, `scripts/f_audit_f2.py`.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = ("1. **CORRECTION I OWE YOU:** my Target F claim that \"all nine concepts died\" is "
       "**WITHDRAWN**. It came from two bugs in my own evaluation, not from the model.")
new = ("1. **CORRECTION I OWE YOU — now re-tested, not just withdrawn:** with the checkpoint rule "
       "fixed, the readable student on generated data scores **R1 5/9, R5 6/9** against Target "
       "E's 7/9 and 6/9 — **the concepts SURVIVE self-training.** The earlier \"all nine concepts "
       "died\" came from two bugs of mine, not the model.")
assert old in txt, "STATUS anchor 1 not found"
txt = txt.replace(old, new, 1)
old5 = ("5. **What still stands — the LM collapse.**")
new5 = ("5. **What still stands — the LM collapse (unaffected by all of this).**")
assert old5 in txt
txt = txt.replace(old5, new5, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
