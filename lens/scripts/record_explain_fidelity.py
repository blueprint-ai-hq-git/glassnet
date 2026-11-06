"""record_explain_fidelity.py — explain() now annotates every concept with its measured fidelity and
refuses to present decorative names as causes. E18 partially closed.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
f = json.load(open(os.path.join(P, "lens", "runs", "seetalk",
                                "seetalk_k22_big_fidelity.json")))
c = f["counts"]

LOG = f"""
---

## 2026-10-09 — FIX SHIPPED: explain() can no longer print a decorative concept name as a cause.

### THE FIX (three parts, all in the shipping interface)
1. **`build_fidelity.py`** measures, for every concept in a checkpoint, how strongly its neuron tracks
   the concept it is NAMED after (`own_corr`), its nearest rival concept (`best_other`), and the
   margin. It writes `<checkpoint>_fidelity.json` beside the checkpoint.
2. **`variants.explain()`** reads that table and annotates every contribution with `own_corr` and a
   verdict. Concepts verdicted **decorative** (own_corr < 0.20) are **excluded from the cause list**,
   and are still reported separately under `decorative_excluded` so nothing is hidden.
3. **No table -> an explicit WARNING** in the note saying the names are NOT verified. Previously the
   note simply asserted the names were the mechanism, which was the false-explanation risk.

### VERIFIED (both cases, real output)
```
  CASE 1 — checkpoint WITHOUT a fidelity table
    fidelity_source: None
    note: "... WARNING: no fidelity table found for this checkpoint, so these names have NOT been
           verified to describe what their neurons detect"          -> PASS
  CASE 2 — checkpoint WITH a fidelity table
    fidelity_source: seetalk_k22_big_fidelity.json
    every listed concept carries own_corr                           -> PASS
    decorative concepts on this checkpoint: 5
    decorative leaked into the cause list : none                    -> PASS
    excluded ones still reported separately: yes                    -> PASS
```
Real annotated output from the shipping checkpoint:
```
    bottom_half_heavier   own_corr 0.891  verdict strong   contrib +1.6036
    is_dark               own_corr 0.976  verdict strong   contrib +0.5820
    has_high_contrast     own_corr 0.848  verdict strong   contrib -0.2763
    bottom_left_mass      own_corr 0.338  verdict weak     contrib -2.1846
```

### GOOD NEWS: THE SHIPPING CHECKPOINT'S CONCEPTS ARE GENUINELY GOOD
```
  seetalk_k22_big:  strong {c.get('strong', 0)}   honest {c.get('honest', 0)}   weak {c.get('weak', 0)}   decorative {c.get('decorative', 0)}
```
**10 strong and 6 honest against only 5 decorative.** The alarming medical result (mean own-corr
0.033 at lam_concept 0.00) does NOT generalise to the shipping artifact. So:
* the shipping see+talk model's concept names are mostly trustworthy — own_corr 0.85-0.98 on the
  strongest ones;
* the decorative-name problem is real but it is a property of WHICH checkpoint you look at, which is
  exactly why the fidelity table is now attached per-checkpoint instead of assumed.

### STATUS OF E18
PARTIALLY CLOSED. Done: (1) every concept now carries a measured fidelity score beside its name;
(3) the AUROC pass bar is now complemented by own-vs-best-other margin and rank. Still OPEN:
(2) the concept set is still heavily redundant (34 correlated pairs on medical), so the layer has
fewer INDEPENDENT concepts than its name count claims — de-duplication is the next piece of work.
Recorded honestly as open, not swept up.
"""

with open(os.path.join(P, "LOG.md"), "a") as f2:
    f2.write(LOG)
print("appended explain fidelity fix record")
print(f"  shipping K22 concepts: {c}")
print("  E18 partially closed (fidelity shipped; de-duplication still open)")
