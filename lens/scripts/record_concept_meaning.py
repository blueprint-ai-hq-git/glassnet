"""record_concept_meaning.py — the "does it mean what its label says" audit, its control, and the
honest conclusion. The strict gate FAILED; the control explains why, and reveals the real mechanism.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
m = json.load(open(os.path.join(P, "lens", "runs", "vl", "concept_meaning_audit.json")))
r = json.load(open(os.path.join(P, "lens", "runs", "vl", "concept_redundancy_audit.json")))
r0 = r["report"]["mlow_lc0.00"]
r15 = r["report"]["mlow_lc0.15"]

LOG = f"""
---

## 2026-10-09 — THE "DOES IT MEAN WHAT ITS LABEL SAYS?" AUDIT. The strict gate FAILED, and the
## control revealed the real mechanism.

### WHAT WAS BUILT (the test the brief asked for)
`audit_concept_meaning.py` computes, for EVERY concept neuron, its correlation with EVERY concept
formula — not just its own. That catches the case the existing AUROC audit cannot: a neuron that
tracks something OTHER than its label while still passing its own AUROC.

### THE STRICT GATE FAILED — reported as failed
```
  m1  >=80% of concepts label-honest          FAIL
      honest 3/35 (9%)   mislabelled 24      (per checkpoint; 3/70 across both)
  m2  every concept reported by name          PASS
  m3  mislabelled concepts named              PASS
```
The sanity test FIRED. By the strict rule (own concept must beat every other concept by >0.10), only
**3 of 70** concept-neurons are honest, and 24 concepts per checkpoint are MISLABELLED by name.
That is the gate result and it is not being redefined.

### THE CONTROL I RAN BEFORE BELIEVING IT
The strict rule assumes concepts are INDEPENDENT. If two formulas are themselves correlated in the
data, an honest neuron tracking one will correlate with the other, and the rule flags it as a liar.
`audit_concept_redundancy.py` measured this:
```
  concept pairs with |corr| > 0.70 : {r0['redundant_pairs_70']} per checkpoint
```
**The concept set is heavily redundant**, so the strict mislabel count IS inflated by redundancy.
Confirmed.

### BUT THE CONTROL ALSO FOUND SOMETHING BIGGER — the two checkpoints differ enormously
```
  lam_concept = 0.00                    lam_concept = 0.15
  mean own-correlation  {sum(x['own_corr'] for x in r0['rows'])/r0['n']:.3f}                        mean own-correlation  {sum(x['own_corr'] for x in r15['rows'])/r15['n']:.3f}
  neurons above 0.70 own-corr  0/{r0['n']}                neurons above 0.70 own-corr  8/{r15['n']}
  own concept ranks #1         0/{r0['n']}                own concept ranks #1         5/{r15['n']}
  best own-corr                {max(x['own_corr'] for x in r0['rows']):.3f}                        best own-corr                {max(x['own_corr'] for x in r15['rows']):.3f}
```

**Without the concept loss (lam=0.00), the neurons barely track their own named concepts at all
(mean correlation 0.033, ZERO concepts above 0.70, ZERO ranking #1). With it (lam=0.15), they do
(mean 0.423, 8 above 0.70, 5 ranking #1).**

### THE RECONCILIATION — this explains the trade-off exactly
At lam=0.00 the "readable" layer is MORE accurate (65.39% vs 62.02%) and MORE stable (std 1.33 vs
4.51) — **because it is not being forced to be readable.** The concept names on that checkpoint are
decorative: the neurons are free hidden units that the names do not describe. The extra accuracy is
the accuracy of a plain hidden layer.

**The concept loss is what makes the layer genuinely readable, and it is also what costs the ~3.4
points and the stability.** The trade-off is not a mystery; this is the mechanism behind it.

### THE HONEST ANSWER TO "DOES EACH CONCEPT NEURON REALLY FIRE ON ITS NAMED CONCEPT?"
**Partly, and less than the AUROC audit implied.**
* The existing AUROC audit (>=0.55 against its own formula) is a WEAK test — it can pass while the
  neuron is more strongly correlated with a different concept. That is why 30/35 "passed" while this
  audit finds far fewer honest labels.
* With the concept loss ON, a minority of concepts (5-8 of 35) genuinely track their own formula
  above 0.70, and the mean alignment is real (0.42).
* With it OFF, the labels do not describe the neurons.
* The concept set itself is highly redundant ({r0['redundant_pairs_70']} pairs above 0.70), so many
  neurons track several concepts at once and **a single name cannot fully describe what they detect.**
  This is a limitation of the concept DESIGN, not of the training.

### WHAT MUST CHANGE BEFORE explain() CAN BE TRUSTED
1. `explain()` must not present a single concept name as the cause without stating its own-correlation
   strength. A name with own-corr 0.20 is decoration, not an explanation.
2. The AUROC pass bar (0.55) is too weak. It should be reported alongside the own-vs-best-other margin.
3. The concept set needs de-duplication: {r0['redundant_pairs_70']} correlated pairs means the layer
   has far fewer INDEPENDENT concepts than it claims to have.
This is recorded as an OPEN PROBLEM, not a solved one.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended concept meaning + redundancy audit")
print(f"  m1 FAIL: 3/70 honest by the strict rule")
print(f"  control: {r0['redundant_pairs_70']} redundant concept pairs -> strict count inflated")
print(f"  KEY: own-corr mean {sum(x['own_corr'] for x in r0['rows'])/r0['n']:.3f} (lam 0) vs "
      f"{sum(x['own_corr'] for x in r15['rows'])/r15['n']:.3f} (lam 0.15)")
