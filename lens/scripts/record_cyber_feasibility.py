"""record_cyber_feasibility.py — the cyber capability: measured feasibility, honest verdict.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — CYBER CAPABILITY: MEASURED, NOT GUESSED. Verdict: NOT VIABLE from the data on disk.

### WHAT WAS TRIED (and why this shape)
Cyber was blocked as "no licence-clean corpus offline". Before writing it off again, I tested whether
a SECURITY-PATTERN CLASSIFIER could be built from data already on disk: the MIT-licensed code corpus
(Magicoder-OSS / codeparrot-apps). Security review is exactly what a transparent concept model is
for — the reasons must be NAMED ("this calls eval", "this builds a raw SQL string") — and no download
or new licence is needed. So: 12 hand-written, human-checkable security concepts, measured over
37,090 documents.

### GATE y1 FAILED — at the standard concept-support bar, only 2 of 12 concepts are learnable
```
  concept              support        verdict
  has_input_read        0.0235        keep
  has_file_multi        0.0776        keep
  has_regex             0.0186        below the 0.02 bar
  has_raw_sql           0.0115        below
  has_network           0.0091        below
  has_shell_spawn       0.0081        below
  has_crypto            0.0066        below
  has_eval_exec         0.0056        below
  has_secret_literal    0.0039        below
  has_sanitize          0.0020        below
  has_pickle_load       0.0018        below
  has_sql_param         0.0017        below
```
**y1 (>= 6 concepts survive) FAILED: 2 survive. y2 PASS (nothing above 0.95). y3 PASS.**
The two survivors are GENERIC (has_file_multi, has_input_read), not security-specific — so even
relaxing nothing, the layer that could be built would not be a security layer.

### A MEASUREMENT BUG WAS FOUND AND FIXED INSIDE THIS CHECK
`has_regex` and `has_pickle_load` first measured **exactly 0**. That was MY bug, not the data: the
corpus is a TOKEN stream, so `re.compile` appears as `re . compile`, and a regex written for source
text never matches. Re-measured with token-aware patterns: has_regex 690 (0.0186), has_pickle 67
(0.0018). Both still below the bar, so the verdict does not change — but a 0 that comes from a
measurement artifact would have been reported as "this pattern is absent", which is false.

### THE VERDICT, AND WHAT IT WOULD TAKE
**No readable security classifier can be built from the corpora on disk.** This is now a MEASURED
statement (12 concepts, 37k docs), not an assumption. The gate was NOT relaxed to make it pass — the
standard 0.02 bar stands and y1 is recorded as FAILED. Rare security patterns are inherently rare in
a general Python problem/solution corpus, which is a property of the data, not of the architecture.

To unblock it honestly requires a security-focused corpus with a clear licence (or the owner's own
data). That is a data-sourcing decision, not a modelling one, and it stays BLOCKED until such a
source exists. Recorded rather than papered over.
"""
with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended cyber feasibility record")
print("  y1 FAILED: 2 of 12 security concepts survive the 0.02 bar")
print("  measurement bug fixed: token-aware patterns (has_regex 0 -> 690)")