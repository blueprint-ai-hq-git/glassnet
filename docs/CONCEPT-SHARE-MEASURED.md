

---

# P1 MEASURED WITH THE CORRECTED INSTRUMENT: THE BOTTLENECK MOVES SHARE BUT NOT SPECIFICITY (2026-11-20 13:15)

The earlier entry on this page recorded P1 as **UNMEASURED** because `train_chat.py` prints no concept-share metric. That is no
longer the state: the pair was re-run with per-arm `--out` tags and measured with the signed instrument
(`tools/chat_explain_causal.py`, now carrying `concept_signed`, `cancellation` and a reconstruct check against the real logit).
This entry SUPERSEDES the "UNMEASURED" line; nothing is edited away above.

## Measured, 96 usable windows per arm (4 windows dropped by the new vocab guard in each arm)

```
                          --direct ON        --direct OFF
signed concept share      88.76%             100.00%
direct term (max)         1.7263             0.0000
named / random ablation   5.39x              4.78x
concepts for 90% margin   22 of 36           22 of 36
cancellation (median)     0.0000             0.0000
reconstruct |error|       1.076              1.083
```

## Verdict, split honestly

- **P1 MET, but only in the trivial sense.** The named layer's share rises 88.76% -> 100.00%. With the shortcut removed there is no
  other path left, so this was guaranteed by construction and proves only that the ablation-derived decomposition is consistent.
- **The informative measure went the other way.** The **named-vs-random** ablation ratio FELL, 5.39x -> 4.78x, and the number of
  concepts needed to remove 90% of the margin is **identical at 22 of 36**. So forcing the answer through the concepts raises their
  *share* without making them *more specifically* responsible. A passenger with 100% of the seats is still a passenger.
- **Caveat that must travel with these numbers:** the reconstruct check does not close - `C + D` differs from the real predicted
  logit by up to **1.076 / 1.083**, so the two-way split is approximate, not exact. Any claim sharper than "the named path carries
  most of the margin" is not supported by this instrument yet.

## What this closes and what it opens

It closes the hope that a bottleneck alone buys readability: share is not specificity. It opens the sharper question the earlier
fired falsifier already pointed at - *which* concepts are load-bearing. 22 of 36 being needed for 90% of the margin, in both arms,
is the number to attack next, not the share.

Note also that the accuracy this run is scoped to is **4.11% vs 3.95%** (20,000 windows, 1 epoch), so it is a direction test on a
small budget, not a headline accuracy.
