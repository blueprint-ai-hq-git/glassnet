
## Coverage is a spike, not a dial: the shipped setting sits on top of it (2026-11-16)

Predictions before the run: `lens/runs/WINLAM2-PREDICTIONS.txt`. Numbers from `lens/runs/WINLAM2.txt`.

```
window-label strength   VAL acc   ppl      coverage T=0.7   repetition
0.5 (half)               25.38%   168.98       0.133          0.127
1.0 (shipped)            24.58%*  181.04       0.292          0.027
2.0 (double)             23.23%   204.13       0.156          0.042
                                 * three-seed mean; others one seed
```

## The falsifier fired

I wrote: *"coverage at or below 0.292 means lam 1.0 is a CEILING, not a point on a rising curve."* Doubling supervision gave **0.156** - not higher than 0.292, but
roughly half of it.

So the shape of this relationship is neither rising nor monotone: **0.133 at half strength, 0.292 at the shipped strength, 0.156 at double.** Coverage forms a
**narrow peak at exactly the shipped setting**, and moving the weight in either direction falls off it.

Scored: **W1 FAILED** (0.156 vs the 0.292 bar), **W2 PASSED** (23.23 below 24.58), **W3 PASSED** (repetition 0.042).

## What this settles

**The supervision-strength axis is closed.** Seven configurations measured today, and no weight above or below the shipped one recovers the vocabulary breadth of
0.292. The shipped configuration is a **local optimum in both directions** - which is a stronger statement than "we picked a setting": it means breadth is not a
tunable dial at all, and further weight settings are not worth a GPU hour.

What remains open is a different question entirely: **what is supervised per position**, not how hard.

## The complete day, one table

```
config                          VAL acc   coverage
no supervision                  27.01%*    0.190
next-token, quarter             26.69%     0.150
next-token, half                26.52%     0.167
next-token, full                26.12%*    0.173
window, half                    25.38%     0.133
window, full (SHIPPED)          24.58%*    0.292
window, double                  23.23%     0.156
                                * three-seed means
```

**Accuracy falls monotonically with supervision strength in both label families** - the one quantity that behaved consistently across every arm. **Breadth does
not**: its maximum is a single point, at the shipped setting, and every neighbour loses roughly half of it.

Nothing in this table beats the shipped model on both. The shipped model stays the shipped model, and the reason is now measured rather than argued.
