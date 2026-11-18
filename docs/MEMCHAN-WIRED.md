# The memory channel is wired into the real trainer, and proven to change nothing (2026-11-18)

`lens/memchan.py` was built and certified in isolation. This wires it into the **actual training step** (`lens/scan_direct_batched.train_step`, the fused forward+backward that every
chat arm uses) and proves the edit is safe on a real checkpoint.

## What was added

1. **A batched form of the channel** (`forward_batched` / `backward_batched`): the trainer runs B samples at once, so each sample needs its **own** bank - `M` is `(B,K,H)`, not
   `(K,H)`. Independent gradients, worst relative error **6.432e-07** against central finite differences, exact no-op at `Wo=0`, and a check that the banks are genuinely **not shared**
   across the batch (a real bug class in batched code).
2. **The hook in the recurrence** at `lens/scan_direct_batched.py:33`, the one line where the state advances:

```
before:  hs[:, t+1] = tanh(xs[:,t] @ Wx + hs[:,t] @ Wh + b)
after:   h_t = tanh(...);  h_t, M_next, cache = memchan(h_t, M_t);  Ms[:, t+1] = M_next
```

guarded by `use_mem = "mem_Wq" in p`, so a checkpoint without the `mem_*` keys takes the original code path unchanged.

## The identity proof, on a real checkpoint (`scan11_r090_ep3000`)

`tools/scan_identity_check.py` runs the real `train_step` with `lr=0` on a fixed batch, and hashes the loss plus **every pre-existing parameter gradient**. The hash deliberately
covers only parameters that existed before, because adding a new parameter necessarily adds its gradient:

```
IDENTITY 1 - no memory keys:
  loss 9.396832162837e+00   pre-existing gradient sha256 c9336cc84854a766ab592832f1555c959e066e8d1f5d52c0d3b329cf77491a83

IDENTITY 2 - memory keys present, mem_Wo = 0:
  loss 9.396832162837e+00   pre-existing gradient sha256 c9336cc84854a766ab592832f1555c959e066e8d1f5d52c0d3b329cf77491a83
  new-parameter max|grad|: mem_Wo=4.870e-03   mem_Wq=0   mem_Wk=0   mem_decay=0
```

**The loss is identical to twelve significant digits and every pre-existing gradient is bit-identical**, with the channel fully present. And the channel is not inert: `mem_Wo` carries a
gradient of **4.870e-03**, which is exactly the term that lets it grow away from the no-op. The other three quarters of the channel receive exactly zero, which is the correct consequence
of `Wo = 0` (no read reaches the output yet, so no query or write can be blamed).

## Two bugs found while wiring, in my own code

1. **`mem_K` was being passed as a parameter.** `train_step` builds its gradient dict from `p.items()`, so an integer `K` in `p` becomes a float "parameter" and the optimiser would
   update it. `K` is now read from `mem_decay.shape[0]` and never passed.
2. **Hashing all gradients was the wrong test.** The first version reported a difference for Identity 2 merely because new parameters appeared in the dict. A correct no-op check has to
   compare the **loss** and the **pre-existing gradients**, not the whole dict - otherwise a perfect no-op looks like a change.

## Status

The component is now **safe to train**: it cannot alter any existing number, and it has the gradients needed to move. The next step is an arm that actually trains it, with the memory
gate (k=16 retention) as the endpoint rather than a hope.
