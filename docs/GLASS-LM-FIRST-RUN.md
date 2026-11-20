

---

# OUR OWN BLOCK LEARNS REAL TEXT: HELD-OUT 12.32% AGAINST A 4.08% MATCHED CONSTANT (2026-11-20 16:05)

The first training run in this project where nothing underneath it is a guess: forward proven against a hand reference, backward
certified against finite differences, the constant baseline built the same way as every other arm.

## Result, 600 steps, 286 seconds, 20,000 real-text windows

```
model                     836,352 parameters in 24 named tensors (E 768,128 + P 2,048 + 2 blocks)
matched constant          4.08%   (most frequent TRAIN target after excluding <unk>: token 3;
                                 605,016 scored train targets, 119,594 scored held-out targets)
HELD-OUT  loss 5.5956     acc 12.32%
delta                     +8.24 points -> BEATS the constant
```

Loss fell 9.4038 -> 5.4656 with no divergence. Train accuracy ended at 12.18%, so held-out (12.32%) is **not** below train - there is
no overfitting signal at this budget.

## Predictions, checked

- **P1 (held-out beats the constant by >= +1.0): MET**, by +8.24.
- **P2 (loss falls without a >1.0 jump between logged checkpoints): MET.** The largest upward step was 5.7026 -> 6.3016 (+0.60).
- **Falsifier not triggered**, so the block is a valid foundation for the next work.

## What this does and does not say

**Does say:** our own attention block, with its own cache, our own LayerNorm, our own momentum SGD and a tied output table, trains on
real dialogue and generalises 8.24 points above a matched constant in under five minutes.

**Does not say:** it is competitive with anything. This is 600 steps at d=64 on 20k windows; the project's other arms ran 6000 steps at
larger widths on the full corpus. No comparison to those is claimed here, and the numbers are not on the same budget.

## The honest flag that became the next job

**E, the token table, is 768,128 of 836,352 parameters = 92%.** That is the same lookup-table dominance the anatomy scan found in the
old architecture (69% there). A model that is 92% table can only memorise; it cannot be "10x smarter per layer" because the layers hold
8% of the capacity. So the next piece is not more training - it is removing the table dominance, starting with a **factored embedding**
(E = A @ B, rank r), which is a real architectural change with a checkable gradient.
