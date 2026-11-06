# RE-AUDIT UNDER METER-02 — old vs new, and the claims it changes

**Date:** 2026-10-04 · **Authority:** owner decision "A + C, as a rule change for the
meter only." · **Bars: UNCHANGED** (AUROC 0.55, contribution 0.25, 50% pass, lie probe
caught) · **Pre-registered in `METER-RULE-02.md` BEFORE any re-audit ran.**

The meter now requires, in addition to the old clauses, that a concept **fire on at
least 2% of validation windows** (activation > 0.5). 1% and 5% are reported as
sensitivity. No model was retrained, re-tuned or re-named for this: it is a re-score
of the same checkpoints, and with `min_firing = 0` the meter reproduces every recorded
Meter-01 pass count **and every per-column flag** exactly (verified: 23/32, 22/32,
19.7/32, 9.3/32 all MATCH).

---

## 1. T2 concept audits

| result | Meter-01 | 1% | **2% (rule)** | 5% | change at 2% |
|---|---|---|---|---|---|
| baseline (unsupervised, lam_w 2.0) | 9.3/32 = 29.2% | 9.3/32 | **9.3/32 = 29.2%** | 8.3/32 | none |
| supervised lc=0.1 | 19.7/32 = 61.5% | 16.3/32 | **9.3/32 = 29.2%** | 5.3/32 | **−32.3 pts** |
| supervised lc=0.5 | 22.0/32 = 68.8% | 19.0/32 | **12.0/32 = 37.5%** | 7.0/32 | **−31.3 pts** |
| supervised lc=1.0 | 23.0/32 = 71.9% | 19.7/32 | **12.7/32 = 39.6%** | 8.3/32 | **−32.3 pts** |
| K=8 selection | 0.7/8 = 8.8% | 0.3/8 | **0.3/8 = 3.8%** | 0.3/8 | −5.0 pts |
| K=16 selection | 2.3/16 = 14.4% | 2.3/16 | **2.3/16 = 14.4%** | 2.3/16 | none |
| K=24 selection | 7.0/24 = 29.2% | 6.7/24 | **6.7/24 = 27.9%** | 6.0/24 | −1.3 pts |
| K=32 selection | 9.7/32 = 30.3% | 9.7/32 | **9.7/32 = 30.3%** | 8.3/32 | none |

Every supervised setting loses ~31–32 points. The unsupervised baseline and the
K=16/K=32 selections barely move, because their passing neurons fire broadly.

## 2. Rungs 1 / 2 / 3

| rung | checkpoint | old | **2% (rule)** | neurons whose verdict changed |
|---|---|---|---|---|
| Fashion-MNIST (rung 1) | `runs/phase2/final_seed0.npz` | 13/14 = 92.9% | **12/14 = 85.7%** | `border_ink` (fidelity 0.769, fires **0.0004**) |
| PathMNIST (rung 2, iter014) | `runs/loop/iter014/final_seed0.npz` | 8/8 = 100% | **8/8 = 100%** | none |
| Banking77 (rung 3, iter018) | `runs/loop/iter018/final_seed0.npz` | 3/16 = 18.8% | **0/16 = 0.0%** | 3 neurons, all firing **0.0000** |

**PathMNIST reproduces the recorded table EXACTLY** (0.646 / 0.978 / 0.974 / 0.986 /
0.973 / 0.569 / 0.717 / 0.968 — all 8 identical to `runs/loop/iter014/audit.md`), so
that old-vs-new comparison is genuinely like-for-like, and "no change" is a real
result rather than a failure to reproduce.

**Banking77 across all three surviving seeds** (`scripts/t3_banking77_seeds.py`):

| seed | pass (old) | pass (2%) | neurons firing on 0.0000 of val |
|---|---|---|---|
| 0 | 3/16 | 0/16 | **all 16** (fidelity up to 0.623) |
| 1 | 3/16 | 1/16 | 14 of 16 (fidelity up to 0.789) |
| 2 | 4/16 | 0/16 | **all 16** (fidelity up to 0.783) |

**On Banking77 essentially every concept neuron never fires on validation data**, yet
the meter was granting passes with fidelity up to 0.789. A fidelity AUROC computed
from a signal that never crosses the firing threshold is not evidence of detection.

---

## 3. CLAIMS THIS CHANGES (flagged explicitly)

1. **"T2 supervised: 71.9% of concepts pass, above the 50% bar."** — **OVERTURNED.**
   Under Meter-02 it is **39.6%**, below the bar. Most of the supervised gain was
   neurons that had effectively stopped firing. The earlier statement that the
   *unsupervised* half (50.0%) was the honest part still holds, and it survives the
   floor at 47.9%.
2. **Rung 3's audit description** ("12 of 16 concepts broken", with `SKEWED-TARGET`
   flags). The **verdict (FAILED / not a win) stands and is unchanged** — but the
   mechanism was different from the description: essentially all neurons were dead,
   so the per-neuron fidelity numbers measured nothing. Any future work on rung 3 must
   fix the dead concept layer before its fidelity numbers mean anything.
3. **Fashion-MNIST's recorded "12/14 neurons pass".** My re-score is on the
   **validation** split (13/14 old → 12/14 new). The recorded audit states it used the
   **held-out test split**, so these are **NOT like-for-like** and I am not presenting
   them as such. A like-for-like re-score would need another logged test touch; I have
   not made one and will not without explicit approval.
4. **PathMNIST 8/8** — unchanged; no earlier claim affected.
5. **T2 baseline 29.2%** — unchanged; its passes were broad (median support 29%).
6. **K=16 14.4%** — unchanged at every floor: its passing neurons all fire ≥5%.

**No earlier claim of success is made worse than it already was** — rung 3 was already
recorded as FAILED, and the T2 supervised "pass" was already NOT MET because the lie
probe failed. What the floor changes is *why* those numbers looked the way they did.

---

## 4. The honest cost, restated

8 of the 16 supervised T2 targets have TRAIN support below 2% (`good` 1.98%, `For`
1.83%, `My` 1.76%, `they` 1.74%, `I'll` 1.66%, `more` 1.55%, `As` 1.51%, `What`
1.49%). A concept that is a *perfect, precise* detector of `What` — firing on exactly
the 1.49% of windows containing it, AUROC 1.0 — is **excluded** by this floor. That
cost is accepted deliberately and recorded in `METER-RULE-02.md`.

## 5. The live lie probe (owner condition 2) — **MISSED**

The probe column now trains **normally** until its validation firing rate reaches 2%,
and only then are its gradients frozen, so it cannot pass merely by being dead.

| run | reached 2% unfrozen? | frozen at | firing at audit | auroc | contrib | verdict |
|---|---|---|---|---|---|---|
| supervised lc=1.0, lr=0.5 | **no** | — | 0.0115 | 0.634 | +16.51 | **INCONCLUSIVE** (caught by the floor only) |
| baseline lc=0, lr=1.0 | yes | epoch 1 | 0.1552 | 0.5591 | +0.4449 | **MISSED** (flag `-`) |

Under supervision the probe column **still dies** (firing 0.0115), so the floor catches
it — but the AUROC/contribution clauses never get tested, which is why the verdict is
INCONCLUSIVE rather than a pass. At baseline the column stays genuinely live (fires on
15.5% of windows) and **passes the meter**, despite having been frozen after a single
epoch — before it could have been trained to detect anything.

**Both pre-registered predictions (METER-RULE-02.md §6) are NOT CONFIRMED.**

### Why — a circularity in the design, not a tuning failure

The namer derives a concept's name **from that column's own firings**, and the answer
layer learns to use **whatever columns exist**. So any live column ends up with:

* a name that matches its firings → AUROC clears 0.55 by construction, and
* a real contribution → the answer layer uses it, so contribution clears 0.25.

The meter therefore asks *"does the name describe the firings?"* — which the namer
guarantees — rather than *"was this concept meant to detect anything?"*. Under this
design a **live** lie probe essentially cannot fail. The floor removed the dead-column
class of false passes; it does not touch this one.

**TARGET A = NOT MET.**

## 6. Files

| what | where |
|---|---|
| the frozen rule | `METER-RULE-02.md` |
| T2 re-scores | `runs/t3/reaudit_t2.json`, `runs/t3/reaudit_all.log` |
| rung re-scores | `runs/t3/reaudit_rungs.json` |
| Banking77 seed check | `scripts/t3_banking77_seeds.py` |
| implementation check | `scripts/t3_meter02_check.py` (floor=0 reproduces Meter-01) |
| memory fix that unblocked this | `scripts/mem_namer.py`, LOG.md 15:45 entry |
| live lie probe runs | `runs/t3/live_probe.log`, `runs/t3/audit_K32_lr*_pa_*_ff002.json` |
