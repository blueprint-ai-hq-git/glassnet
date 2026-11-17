
## CORRECTION: the memory is imposed at 0.9 by the trainer, not chosen by the gradient - and radius alone cannot buy it back (2026-11-17)

The previous entry concluded that "gradient descent had no reason to keep memory and a cheaper short-horizon solution available". **That explanation is wrong.** The decay is
imposed by an explicit renormalisation the trainer runs after **every step**:

```python
# lens/scan_batched.py:54
def renorm_recurrent(p, target=1.0):
    """Keep the recurrent loop stable: rescale Wh to a target spectral radius."""
    r = float(np.max(np.abs(np.linalg.eigvals(W)))
    p["Wh"] = W * (target / r)

# lens/scripts/train_scan9.py:46 - called every step
renorm_recurrent(p, TARGET_RADIUS)      # TARGET_RADIUS = 0.9 from run_scan9.sh
```

A spectral radius of **0.9 is below 1**, so every step contracts a perturbation by construction. The checkpoint's measured radius is exactly **0.9000**. Memory was not spent by the
optimiser; it was **clamped** by the trainer after each update. The init-vs-trained contrast still stands as a measurement (17.81% -> 0.858%), but the cause is this line, not a
short-horizon preference.

## The radius sweep, no training, existing checkpoint

```
radius     k=16 shift        acc%    entropy
   0.9         0.703%       17.84%     6.6152
   1.0         1.864%       18.68%     6.7658     <- best accuracy
   1.2         1.241%       11.26%     8.1006
   1.5         0.759%        4.88%     8.1733
   2.0         1.256%        3.65%     8.4291
   2.5         3.185%        2.73%     8.4368
   3.4        15.380%        1.50%     8.6826     <- memory gate passed, accuracy destroyed
```

## Two results

**1. A free accuracy gain exists at radius 1.0: 18.68% against 17.84% at the shipped 0.9.** Nearly a full point, obtained by one scalar rescale of the existing weights, no training. That
is worth having and it should be re-measured properly (more tokens, and on a trained arm rather than a post-hoc rescale) before it goes on any page as an improvement.

**2. The gate cannot be reached by radius alone.** The memory gate (k=16 above 10%) **is** reachable - at radius 3.4 the shift is **15.380%** - but there accuracy collapses to **1.50%** and
entropy rises to 8.68 nats, which is the tanh saturation the function's own docstring documented ("at radius 3.4 the loop saturates tanh and every prompt produced the same state, making the
model input-blind"). Between 1.0 and 2.5 the shift does not rise at all while accuracy falls off - so this scalar does not trade smoothly, it works only at the extremes.

**Conclusion: a single scalar on the recurrent transition cannot satisfy both conditions on these weights.** Memory without saturation needs a mechanism that separates "keep information"
from "amplify everything", which is precisely what a gate does and what a global rescale cannot.

## What stands, corrected

- **Measured and unchanged:** state is load-bearing (-16.68 points), order-sensitive (-11.04), effective memory ~single-digit tokens, decay half-life ~1.5 tokens.
- **Corrected:** the cause is `renorm_recurrent(..., 0.9)` applied every step, not an optimiser preference.
- **New and verified:** radius 1.0 gives 18.68% vs 17.84% (post-hoc rescale, needs confirmation); radius alone reaches the memory gate only where accuracy dies.
- **Still the right lever:** gating, and a loss term that pays for retention. Both now have a clear opponent to beat - the 0.9 clamp - which any such mechanism must be tested **with the clamp
  raised or removed**, or it will be silently normalised back to the same behaviour.
