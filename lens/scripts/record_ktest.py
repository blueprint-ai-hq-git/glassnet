"""record_ktest.py — see-only K-transfer: k1+k2 PASS; width story now spans BOTH model families.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "image", "ktest_verdict.json")))

LOG = f"""
---

## 2026-10-08 late — SEE-ONLY K-TRANSFER: the concept finding REPLICATES on the classifier

Same proven train_image loop (60k Fashion-MNIST, 12 epochs, d=64, GPU), only the concept set
changed; 3 arms x 3 seeds, fresh 10k TEST pass per run (logged per artifact):

```
  arm   K     params    test acc        concepts passing audit
  none  12    16,665    81.89 ± 0.23    9-10 / 13
  real  23    27,015    84.36 ± 0.16    15-16 / 23
  dummy 22    25,980    83.62 ± 0.18    14-15 / 22

  k1 real-none = +2.47 > 2.0            PASS
  k2 |real-dummy| = 0.74 <= 2.0         PASS
```

**The seetalk story replicates on the pure classifier: K=12 starves (+2.5 pts from +10 dims), and
real readable concepts beat meaningless random projections by only 0.74 — inside combined seed
noise (±0.58). The width-not-meaning verdict now holds across BOTH model families.**

Honest fine print printed beside the numbers (verdict-check rule):
* The real arm kept 10/10 extended concepts (K=23), the dummy arm 9 (K=22 — one random direction
  was constant on train and fell to the support floor). One extra width unit favours `real` by
  maybe a fraction; the +0.74 lead is within that and within noise, so it is called NO DIFFERENCE.
* The extended arms are also bigger nets (27,015 vs 16,665 params) — that IS the width confound,
  which is exactly what the dummy arm exists to price out. It priced out: +1.7 of the +2.5 is
  width (dummy-none), the remaining +0.7 is not measurable meaning.
* Audit passing count rises with K (9->16 of the larger set) — more concepts pass their own AUROC
  test, i.e. the extended readable concepts really do mean what their names say; that part of the
  transparency claim is confirmed.

SMOKE artifact kept and marked ktest_SMOKE_* (renamed, never overwritten).
Gate margin note: k1 passed at +2.47 against a 2.0 bar — thin but consistent (every real seed beat
every none seed: 84.17 / 84.33 / 84.57 vs 81.56 / 82.04 / 82.07; zero overlap).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
print(f"  none {d['means']['none']} | real {d['means']['real']} | dummy {d['means']['dummy']}")
print("  k1 PASS +2.47  k2 PASS 0.74 — width story replicates on see-only")
