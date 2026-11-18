# The 8-slot memory bank was one slot in disguise, provably (2026-11-18)

The separate memory channel was built to give the model somewhere to store information that does not have to survive a contracting recurrent matrix. Its first arm is 500 steps into a
1500-step run, and the early reading is small:

```
k=16 memory   1.605%   (resumed baseline 1.312%)   +0.293 points, against a 10% gate
held-out      21.03%   (baseline 21.03%)            no accuracy cost
loss          6.3199 nats (baseline 6.4417)         -0.12 nats, better
```

So P1 (memory rises above 1.322%) is met, but trivially, and P2 (no accuracy cost) is holding. The number that matters is the one P3 was written to expose, and it exposed a defect:

```
mem_decay: 0.8580580949783325  0.8580580949783325  0.8580580949783325  ...  (all 8 identical, bit for bit)
```

## Why the slots can never differentiate

This is not convergence, it is exact symmetry, and it is easy to see in the forward pass:

1. `M` starts at **zero** for every slot.
2. Every slot therefore scores `q . M = 0`, so the softmax over slots is **uniform**.
3. Every slot receives the **same** write `a[b,k] * k[b]`.
4. So `M` stays identical across slots, the scores stay identical, and `d_decay[k]` is identical for every `k`.

Nothing in that loop breaks on its own. **The 8-slot bank is mathematically ONE slot with 8x the write rate**, which is exactly what a memory gain of 0.293 points looks like.

## The fix, and the arm that tests it

`MEM_SYM=1` (now the default for new runs) initialises the per-slot decay from `uniform(0.85, 0.95)` instead of all `0.9`, so the slots start different and their gradients differ.
`MEM_SYM=0` restores the old behaviour bit-for-bit, so every earlier arm stays reproducible.

`lens/scripts/run_memchan_sym.sh` re-runs the identical recipe with that one change, queued behind the running arms:

- **P1** the slots differentiate: max-min across the 8 learned decays **> 0.005** (the symmetric arm scored **0.000000**)
- **P2** k=16 memory exceeds **1.605%**, the symmetric channel's reading
- **P3** no accuracy cost: held-out at or above the symmetric arm's own reading at the same protocol
- **FALSIFIER** decays still identical, or k=16 at or below 1.605% at the endpoint. Then symmetry was **not** the limiter, and the earlier falsifier branch stands: the limit is the
  **training signal** - a 32-token window cannot teach what must survive 16 steps - and the next question is the window, not the bank.

## Two robustness bugs found while editing, and fixed

The editor's type checker flagged both, and both were real:

1. **`Mm` was possibly unbound** in the held-out path - it was only assigned inside `if _MEM is not None`, then used inside the loop. Safe today, a crash waiting for the next refactor. Now
   bound unconditionally to `None` first.
2. **`ema` could be `None`** in the final log line, so a run that trained **zero** steps would crash with `TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'` - the same
   zero-step class of failure that was already fixed once in the loop. Now defaulted.

A fresh 2-step run completes end to end after the fixes (`HELD-OUT loss 9.7103 acc 0.00%`, `smoke_fix COMPLETE`), so the paths the fixes touched are exercised.
