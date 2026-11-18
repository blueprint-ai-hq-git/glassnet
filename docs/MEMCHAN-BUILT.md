# A separate memory channel, built and certified as a no-op (2026-11-18)

The memory line reached a measured conclusion: **a recurrent weight matrix cannot deliver the 10% k=16 gate.** Reshaping `Wh` walks a trade frontier and stops there - radius 0.9 1.322%
(21.02% accuracy), radius 1.0 4.434% (20.83%), radius 1.05 6.695% (11.90%), radius 1.10 8.511% (10.05%), orthogonal projection 7.888% (13.34%). So memory now gets its **own
channel** instead of being squeezed through the recurrent matrix.

## What was built

`lens/memchan.py` - a K-slot memory bank `M (K,H)` carried across timesteps:

```
q      = h @ Wq                     read query
s      = q @ M.T / sqrt(H)          match every slot
a      = softmax_k(s)               attention over slots
read   = a @ M                      retrieved vector
k      = tanh(h @ Wk)               write key (rank-1 per slot)
M_next = decay * M + a.T @ k        learned per-slot decay + key-addressed write
out    = h + tanh(read @ Wo)        the channel's contribution
```

The point of the design is that the **growth** a memory needs (more paths to fit) is decoupled from the **forgetting** that `Wh` could only express by contraction. Decay is per-slot and
learned; the recurrent matrix is left alone.

## Verified as a no-op, by construction and by measurement

`Wo` is initialised to **zero**, so `tanh(read @ 0) = 0` and `out == h`. The self-test measures it rather than assuming it:

```
1) no-op at init: max |out - h| = 0.000e+00   (exactly zero, not merely small)
```

So this channel can be injected into any existing run without changing a single number at step 0 - the project's standing rule for a new component.

## Gradients, certified

Hand-written backward against central finite differences (`tools/memchan_selftest.py`), including the cross-timestep term (`d_M_out` is a real input, because the loss touches every
timestep):

```
   param    max abs diff   max rel diff   result
   Wq       1.899e-09      1.102e-08      PASS
   Wk       5.379e-10      2.704e-09      PASS
   Wo       1.094e-09      1.689e-08      PASS
   decay    5.682e-10      2.097e-10      PASS
   dh       8.332e-10      6.391e-09      PASS
   dM       1.440e-09      2.963e-09      PASS

WORST RELATIVE GRADIENT ERROR: 1.689e-08   -> MEMCHAN SELF-TEST: PASS
3) with Wo non-zero: max |out - h| = 0.6743   (it can act)
4) memory bank moves per step: max |M_next - M| = 0.9732
```

## Two real bugs, caught by the checks rather than by reading

1. **A transposition in the write path.** `d_a += d_M_out @ k.T` produced `(K,B)` where `(B,K)` was required. It failed loudly with a broadcast error - the cheap kind.
2. **A missing gradient path that the parameter checks could not see.** `read = a @ M` gives `M` a *direct* path into the output, and `dM` was missing the `a.T @ d_read` term. **Every
   parameter gradient still passed** - `Wq`, `Wk`, `Wo`, `decay`, and `dh` were all at 1e-09 - while `dM` was wrong with a relative error of **1.000** (one side zero, the other not).
   Only checking the **input** gradient found it.

That second one is the lesson worth keeping: for a component whose state *is* an input, checking parameter gradients is not enough. The state gradient has to be checked too, or a
wrong memory update ships with clean-looking weights.
