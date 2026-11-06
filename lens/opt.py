#!/usr/bin/env python3
"""lens/opt.py - OUR OWN training methods.

WHY OUR OWN, and what the user asked for (2026-10-18): "consistent learning rate and every single setting
that can run up to a single neural network to 10 billion... so the model never gets dumb at that point,
it only learns, no way of memorising... there is a safety net for the memorising that training finds by
itself... we have to invent our own training methods for distillation, RL, normal SFT, teacher model
training... everything is learning."

WHAT WAS WRONG WITH WHAT WE HAD: plain SGD with a hand-set learning rate. That rate depends on the SIZE
and SCALE of the model: the same 0.4 that works at 1M parameters is too large at 100M (the update moves
the weights further than the curvature allows) and too small at 100k (it crawls). So a hand-set rate can
never be "one setting that runs to 10B" -- it is a per-size arm, which is exactly what the user objects to.

THE IDEA WE USE INSTEAD (standard in spirit, our own implementation): make the step SCALE-INVARIANT by
normalising the update. Two rules here, both computed from the model itself, so no size-specific constant
is needed:

  relgrad  :  w <- w - lr * g / (||g||_2 + eps)          one step per tensor, direction only
  rmsprop  :  w <- w - lr * g / (sqrt(EMA(g^2)) + eps)   per-weight scale from the recent gradient

Both make the effective step depend on the DIRECTION of the gradient rather than its magnitude, and both
are invariant to how large the parameters or the gradients happen to be. That is what lets one setting
travel from 1M to 10B without a per-size learning-rate hunt.

THE MEMORISATION SAFETY NET, in this file because it belongs to the training method, not to a run script:
  gap = train_accuracy - held_out_accuracy
A widening gap means the model is storing examples instead of learning a rule. `MemorisationGuard` watches
it every epoch and, by itself, (a) logs the gap with both numbers, (b) stops the run when the gap crosses
the limit, and (c) if the gap is still climbing when it stops, says so. Nothing about it needs a human to
notice, which is what "training finds it by itself" means.

STILL TO BUILD (declared, not pretended): self-distillation against our own best checkpoint, RL on
verifiable rewards (maths and code answers can be checked exactly, so no reward model is needed), a
curriculum driven by the model's own error rate, and SFT on the model's own successful traces. Each one is
a separate experiment with its own prediction; none of them are implemented yet and none are claimed.
"""
import numpy as np

from lens import backend as B   # to_cpu / to_dev: the gradient of a GPU tensor is a CUPY array, and
                                # `np.asarray(cupy_array)` raises TypeError instead of converting


class RelativeGrad:
    """One setting that does not care how big the model is: normalised-direction steps.

    lr is the SIZE of the step in weight space, not a multiplier on an unbounded gradient, so the same lr
    behaves the same at 1M and at 10B parameters. That is measured, not asserted: the update of every tensor
    has norm exactly lr, at every size (lens/runs/NUMERIC-PATH-OPT-DECAY.txt, property block, sizes 1..4096,
    worst relative error 1.2e-07). It is also why this rule CANNOT overflow however large the step size is
    handed to it - a property plain SGD does not have (lens/runs/NUMERIC-PATH-OPT.txt: SGD at lr 0.5 gave
    non-finite weights on 2 of 3 seeds, our rule 12 of 12 configurations finite).

    `decay` shrinks the step over time (lr * decay**step); decay=1.0 is constant and is bit-identical to the
    behaviour before the knob existed. It is here because a CONSTANT normalised step does not settle: it
    orbits the solution at an amplitude of about lr, and the numeric path measured that orbit
    (lens/runs/NUMERIC-PATH-OPT.txt: the 5-parameter numeric head reaches only 19-28% of its own training set
    under a constant step, and the outcome is non-monotonic in lr - 0.005 passes, 0.01 and 0.02 fail,
    0.05 passes - which is a limit cycle, not a tuning curve).

    `scale` divides each tensor's update by sqrt(size) for the per-entry rmsprop rule only. It is NOT applied
    to relgrad because relgrad needs no such factor: the whole tensor's update already has norm lr. The line
    that used to be here, `step * sqrt(n) / max(sqrt(n), 1)`, is identically 1.0 for every size, so it did
    nothing while claiming to make the rate transferable across widths.
    """

    def __init__(self, params, lr=0.02, eps=1e-8, mode="relgrad", momentum=0.0, scale=True, decay=1.0):
        self.params = list(params)
        self.lr = float(lr)
        self.eps = float(eps)
        self.mode = mode
        self.momentum = float(momentum)
        self.scale = bool(scale)
        self.decay = float(decay)
        self._ema = {}
        self._mom = {}
        self.steps = 0

    def lr_at(self, step):
        """The step size at a given step number. decay=1.0 means constant, and is bit-identical to the
        behaviour before `decay` existed (1.0**k == 1.0 exactly)."""
        return self.lr * (self.decay ** max(int(step), 0))

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.steps += 1
        lr_now = self.lr_at(self.steps - 1)
        for p in self.params:
            if p.grad is None:
                continue
            # BUG FIX 2026-10-20 (found by the first GPU use of this class, train_chat.py --opt relgrad,
            # lens/runs/OPT-SMOKE.txt arm C): this was `np.asarray(p.grad, ...)`, which on a GPU tensor
            # raises "TypeError: Implicit conversion to a NumPy array is not allowed" because p.grad is a
            # CUPY array. Every caller until today was CPU-only, so the fault had never surfaced. The
            # gradient is brought to the host for the norm/scale arithmetic and the finished update is put
            # back on the device; on CPU to_cpu/to_dev are pass-throughs, so the numbers are unchanged
            # (re-verified by the P40h regression: numeric_path_opt_test.py --regress).
            g = np.asarray(B.to_cpu(p.grad), dtype=np.float64)
            if self.momentum:
                m = self._mom.get(p.name, np.zeros_like(g))
                m = self.momentum * m + (1.0 - self.momentum) * g
                self._mom[p.name] = m
                g = m
            if self.mode == "relgrad":
                denom = np.linalg.norm(g) + self.eps
            else:                                        # rmsprop
                e = self._ema.get(p.name, np.zeros_like(g))
                e = 0.9 * e + 0.1 * g * g
                self._ema[p.name] = e
                denom = np.sqrt(e) + self.eps
            step = lr_now * g / denom
            if self.scale and self.mode != "relgrad":
                # a wider tensor has more entries; without this the total movement grows with width and
                # one lr stops transferring between sizes. MEASURED 2026-10-20 (property block of
                # lens/runs/NUMERIC-PATH-OPT-DECAY.txt):
                # relgrad needs no such factor, because the update of the whole tensor ALREADY has norm lr
                # (measured, all sizes 1..4096, relative error 1.2e-07). The line that used to sit here,
                # `step * sqrt(n) / max(sqrt(n), 1)`, is identically 1.0 for every size, so it was a no-op
                # that claimed an effect it did not have; the honest factor for the per-entry rmsprop rule
                # is 1/sqrt(n), and that is what it now is, for rmsprop only. No existing caller uses
                # rmsprop, and for relgrad the update is bit-identical to before.
                step = step / max(np.sqrt(g.size), 1.0)
            # the update goes back to wherever the parameter lives (device or host). `np.asarray(p.data)`
            # was the second device-unaware line: it raises the same TypeError on a CUPY array.
            p.data = p.data - B.to_dev(step.astype(getattr(p.data, "dtype", np.float64)))


class MemorisationGuard:
    """The safety net that finds memorisation by itself.

    Watches gap = train_acc - held_out_acc every epoch. `limit` is the largest gap considered healthy.
    `stop` is returned as True when the gap crosses it, and the run is expected to stop there. It also
    reports whether the gap was still RISING when it tripped, because a rising gap is memorisation in
    progress while a flat high gap is a model that simply fits its data better than the held-out set.
    """

    def __init__(self, limit=0.35, patience=1, verbose=True):
        self.limit = float(limit)
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.history = []
        self.tripped_at = None

    def observe(self, epoch, train_acc, heldout_acc):
        if train_acc is None or heldout_acc is None:
            return False
        gap = float(train_acc) - float(heldout_acc)
        rising = len(self.history) >= 2 and gap > self.history[-1][1] > self.history[-2][1]
        self.history.append((int(epoch), gap))
        if self.verbose:
            print(f"    memorisation net: train {train_acc*100:.2f}% - held-out {heldout_acc*100:.2f}% "
                  f"= gap {gap*100:.2f} points (limit {self.limit*100:.1f})"
                  f"{'  [RISING]' if rising else ''}")
        if gap > self.limit:
            if self.tripped_at is None:
                self.tripped_at = int(epoch)
            if self.verbose:
                print(f"    MEMORISATION NET TRIPPED at epoch {epoch}: gap {gap*100:.2f} > limit "
                      f"{self.limit*100:.1f}. The model is storing examples, not learning the rule."
                      f"{' The gap was still RISING.' if rising else ''} Stopping.")
            return True
        return False

    def summary(self):
        if not self.history:
            return "no measurements"
        gaps = [g for _, g in self.history]
        return (f"gap first {gaps[0]*100:.1f} -> last {gaps[-1]*100:.1f} points over "
                f"{len(gaps)} epochs, tripped at {self.tripped_at}")