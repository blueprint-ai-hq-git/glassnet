"""lens/sentstate.py — the RUNNING SENTENCE STATE ("what I am saying"), the first half of
ARCHITECTURE-FIX.md item C, as a standalone module that nothing imports by default.

WHY THIS EXISTS (measured, not asserted)
----------------------------------------
Every arm this project has trained loses the thread after one token of its own output — free-running
step-8 accuracy 1.6-3.9% against teacher-forced 20.7-25.0% (lens/runs/TALK-FAIR-*.txt), 0 of 6 topical
answers on every ship candidate (PADFREE-TALK-*.txt), and the diagnosis in
lens/runs/DECISION-SHAPE-SCALE.md section 1 is that the model has the MARGINAL shape of its data and not
the CONDITIONAL one. The answer layer reads only the 36 named concept values of the current window plus a
position-summed direct path: there is no carried state of what the model is in the middle of saying.
That is the owner's own item C ("a running sentence state - WHAT I AM SAYING - carried across positions,
made of named concepts so it is readable, and fed to the answer layer beside the token-level
representation", ARCHITECTURE-FIX.md) and it is the only structural change on the project's list that has
never been built.

THE RECURRENCE
--------------
    s_t = tanh( h_t @ Wc + s_{t-1} @ Ur + b )        h_t = the model's own concept vector (K values)
    answer input = concat([ h_t , s_t ])             the answer layer sees concepts AND the state

DESIGN CONSTRAINTS THAT ARE NOT NEGOTIABLE HERE
-----------------------------------------------
* ENGINE-EXPRESSIBLE. lens/tensor.py has no slicing of a 3-D tensor and no 3-D transpose, so the carry
  over positions is a REGISTERED OP (`sstate_scan`) with a hand-written reverse-time backward — the same
  pattern the project already uses for concept_bce / concept_mse / l1_act in lens/model.py. That
  backward is finite-differenced by the pre-flight (lens/runs/SENTSTATE-SCAN-STRUCTURAL.txt), never
  trusted, because a wrong reverse-time recursion trains a plausible-looking model that learns nothing.
* READABLE BY CONSTRUCTION. The state is written by the model's own NAMED concepts through Wc, so every
  state value is an exact linear function of named concept values and can be attributed per concept
  (`explain_state`). No unlabelled head, no supervision this corpus does not have.
* TRUNCATED CARRY. `s_{t-1}` is passed in detached and a tensor that requires grad is REFUSED by name.
  A gradient that crosses a training step is silent coupling; this project has been burned by silent
  inert levers before (a gate that never fired, a concept loss that was never active).
* NOTHING IS ADDED UNLESS ASKED FOR. This module is imported by nothing by default: lens/model.py,
  lens/scripts/train_chat.py and lens/decode.py are NOT touched by it, so every stored checkpoint keeps
  loading and every stored number keeps its meaning.
* COST AT THE SHIP CONFIG (K=36, ds=64, V=12,002): Wc 2,304 + Ur 4,096 + b 64 = 6,464 parameters, and
  the answer layer's input grows from 36 to 100, i.e. +V*ds = 768,128 parameters. +13.11% on the
  5,908,686-parameter d=256 ship, total 6,683,278 — far inside the 50M cap.
"""
import numpy as np

from lens import backend
from lens import tensor as T


class SentenceState:
    """The carried state. Deliberately tiny and deliberately dumb: three parameters, one tanh.

    `ds` (the state width) is the only knob. It is NOT a second concept layer: the concept vector h is
    what goes IN, so the state can only ever carry combinations of concepts the model already names.
    """

    def __init__(self, K, ds=64, rng=None, name="sstate"):
        self.K, self.ds = int(K), int(ds)
        if self.K < 1 or self.ds < 1:
            raise ValueError(f"K and ds must be >= 1, got K={self.K} ds={self.ds}")
        rng = np.random.default_rng(0) if rng is None else rng
        # Small He-style init: an untrained carry must NOT swamp the concept path on step 1, otherwise a
        # 1-epoch arm measures the initialisation instead of the lever.
        sc_w = (2.0 / self.K) ** 0.5
        self.Wc = T.Tensor(backend.to_dev((rng.standard_normal((self.K, self.ds)) * sc_w)
                                          .astype(np.float32)),
                           requires_grad=True, name=f"{name}.Wc")
        # CARRY INIT IS A DECAYED IDENTITY, AND THAT IS A MEASURED FIX, NOT A TASTE (P111e FAIL, attempt 1
        # of lens/runs/SENTSTATE-STRUCTURAL.txt): a zero-mean random Ur at scale 1/sqrt(ds) is NOT
        # contractive — one silent step (h=0) took a state of 0.9 to 0.9910, i.e. it GREW, and 64 steps of
        # random input saturated tanh at 1.0000. A recurrent state that amplifies at initialisation either
        # explodes over a long conversation or saturates to a constant, and both look like "the lever did
        # nothing" in a 6-epoch arm. With Ur = 0.9*I + tiny noise the carry is a learned per-unit decay:
        # it remembers (0.9 per step) and it provably shrinks (|tanh(0.9*0.9)| = 0.67 < 0.9), so the state
        # stays informative instead of pinning at the rails. The diagonal is still trainable.
        u0 = np.eye(self.ds, dtype=np.float64) * 0.9
        u0 += rng.standard_normal((self.ds, self.ds)) * 0.01
        self.Ur = T.Tensor(backend.to_dev(u0.astype(np.float32)),
                           requires_grad=True, name=f"{name}.Ur")
        self.b = T.Tensor(backend.to_dev(np.zeros(self.ds, dtype=np.float32)),
                          requires_grad=True, name=f"{name}.b")
        self.name = name

    # ---------------------------------------------------------------- parameters / state
    def parameters(self):
        return [self.Wc, self.Ur, self.b]

    def initial(self, n):
        """The state at the start of a conversation: zeros, in the engine's dtype, on the host.

        Host-side on purpose: the trainer keeps one state per conversation row across steps and must be
        able to slice/assign it with numpy without a device round-trip.
        """
        return np.zeros((int(n), self.ds), dtype=np.float32)

    # ---------------------------------------------------------------- forward
    def __call__(self, h, s_prev):
        """(N, K) concept vector + (N, ds) previous state -> (N, ds) new state.

        `s_prev` must be DETACHED (numpy array, or a Tensor with requires_grad=False). A tensor that
        requires grad is refused by name: the carry is truncated at the training step by construction.
        """
        if isinstance(s_prev, T.Tensor):
            if s_prev.requires_grad:
                raise ValueError(
                    "SentenceState: s_prev has requires_grad=True. The carry is TRUNCATED: pass the "
                    "previous state detached (numpy array or a no-grad tensor). Backpropagating through "
                    "a training-loop state is silent coupling across steps, which this module refuses "
                    "by design rather than by convention.")
            s = s_prev
        else:
            s = T.Tensor(backend.to_dev(np.asarray(s_prev, dtype=np.float32)), requires_grad=False)
        if s.data.shape != (h.data.shape[0], self.ds):
            raise ValueError(f"SentenceState: s_prev must be (N={h.data.shape[0]}, {self.ds}), "
                             f"got {tuple(s.data.shape)}")
        hp3 = T.reshape(h, (h.data.shape[0], 1, self.K))
        return sstate_scan(hp3, self.Wc, self.Ur, self.b, s_init=s)

    def answer_input(self, h, s):
        """What the answer layer reads when the lever is on: named concepts AND the carried state."""
        return T.concat([h, s], axis=1)                    # (N, K + ds)

    # ---------------------------------------------------------------- readability
    def explain_state(self, h_row, s_row, concept_names=None, top=5):
        """Host-side attribution of ONE state vector to NAMED concepts. Exact, not approximate.

        With s = tanh(Wc^T h + ...), the linear pre-activation of state unit j is sum_k h_k * Wc[k, j],
        so the contribution of concept k to state unit j is exactly h_k * Wc[k, j]. This returns, per
        concept, the L2 size of its contribution across the state — i.e. WHICH NAMED CONCEPTS ARE
        WRITING THE STATE — which is the readability property item C asks for, with no extra training.
        """
        h_row = np.asarray(h_row, dtype=np.float64).reshape(-1)
        Wc = np.asarray(backend.to_cpu(self.Wc.data), dtype=np.float64)
        if h_row.size != self.K:
            raise ValueError(f"h_row must have K={self.K} values, got {h_row.size}")
        contrib = h_row[:, None] * Wc                      # (K, ds) exact per-concept contribution
        mag = np.linalg.norm(contrib, axis=1)
        order = np.argsort(-mag)[:int(top)]
        names = concept_names or [f"concept_{k}" for k in range(self.K)]
        return [(str(names[int(k)]), float(h_row[int(k)]), float(mag[int(k)])) for k in order]


# ======================================================================================
# THE POSITION SCAN — the kernel that makes the carry usable in THIS trainer
# ======================================================================================
# WHY IT EXISTS: the trainer samples RANDOM windows from the 16,878,132-window set (a permutation every
# epoch), so there is no "previous window" whose state could be carried: a state carried across shuffled
# windows carries noise. The carry has to run over the 32 POSITIONS INSIDE the window, and the answer
# layer reads the state after the last position. The engine cannot express that recurrence — lens/tensor.py
# has no slicing of a 3-D tensor and no 3-D transpose, only reshape/concat/matmul/bmm3/add/mul — so it is
# a REGISTERED OP with a hand-written reverse-time backward, exactly like concept_bce/concept_mse/l1_act
# in lens/model.py. Predictions P112a-f (lens/runs/SENTSTATE-SCAN-PREDICTIONS-P112.txt) were written
# BEFORE this code, and the pre-flight finite-differences every parameter it touches.

class PosConcepts:
    """Per-TOKEN concept firing: hp_t = sigmoid(E[x_t] @ Wc2), the same NAMED concepts as the window
    layer but fired at every position. This is what the running state is written from, which is what
    keeps the state readable ("made of named concepts", ARCHITECTURE-FIX.md item C).

    Cost at the ship config: D*K = 256*36 = 9,216 parameters.
    """

    def __init__(self, D, K, rng=None, name="pconcept"):
        self.D, self.K = int(D), int(K)
        rng = np.random.default_rng(0) if rng is None else rng
        scale = (2.0 / self.D) ** 0.5
        self.Wc2 = T.Tensor(backend.to_dev((rng.standard_normal((self.D, self.K)) * scale)
                                           .astype(np.float32)),
                            requires_grad=True, name=f"{name}.Wc2")

    def parameters(self):
        return [self.Wc2]

    def __call__(self, x_rows):
        """x_rows: (N*W, D) per-token representation -> (N*W, K) per-token concept probabilities."""
        return T.sigmoid(T.matmul(x_rows, self.Wc2))


def sstate_scan(hp3, Wc, Ur, b, s_init=None):
    """s_t = tanh(hp_t @ Wc + s_{t-1} @ Ur + b) for t = 0..W-1; returns s_{W-1}.

    hp3: (N, W, K) per-position concept vectors. `s_init` (optional, DETACHED, (N, ds)) seeds the
    recurrence — that is what a generation needs (the state of the conversation so far) and it is also
    how the single-step form SentenceState.__call__ is implemented, so there is ONE tanh in this file
    and not two: P112d measured a 2.2e-16 disagreement between `xp.tanh` and the composed
    `2*sigmoid(2x)-1`, which is algebraically nothing and structurally a second implementation waiting
    to drift (BUGS.md [BUG-SENTSTATE-TWO-TANH-PATHS]).

    Forward is a plain loop over W (32 steps of (N, ds) work — trivial next to the model's own
    matmuls); backward is the reverse-time recursion and is finite-differenced by the pre-flight.
    """
    xp = backend.xp
    if hp3.data.ndim != 3:
        raise ValueError(f"sstate_scan expects (N, W, K), got {tuple(hp3.data.shape)}")
    n, wt, _k = hp3.data.shape
    ds = Ur.data.shape[0]
    if s_init is None:
        s = xp.zeros((n, ds), dtype=hp3.data.dtype)
        parents = (hp3, Wc, Ur, b)
    else:
        if s_init.requires_grad:
            raise ValueError(
                "sstate_scan: s_init has requires_grad=True. The carry is TRUNCATED: pass the initial "
                "state detached (numpy array or a no-grad tensor).")
        if s_init.data.shape != (n, ds):
            raise ValueError(f"sstate_scan: s_init must be (N={n}, ds={ds}), "
                             f"got {tuple(s_init.data.shape)}")
        s = s_init.data
        parents = (hp3, Wc, Ur, b, s_init)
    s_start = s
    states = []
    for t in range(wt):
        a = hp3.data[:, t, :] @ Wc.data + s @ Ur.data + b.data
        s = xp.tanh(a)
        states.append(s)
    rg = any(p.requires_grad for p in parents)
    return T.Tensor(states[-1], rg, _parents=parents, _op="sstate_scan",
                    _saved={"hp3": hp3.data, "states": states, "s_start": s_start})


def _bw_sstate_scan(out, parents, saved, grad):
    """Reverse-time backward for the scan.

    dL/da_t = dL/ds_t * (1 - s_t^2)          with dL/ds_t = grad_t + dL/da_{t+1} @ Ur^T
    dL/dWc += hp_t^T @ dL/da_t               dL/dUr += s_{t-1}^T @ dL/da_t
    dL/db  += sum_N dL/da_t                  dL/dhp_t = dL/da_t @ Wc^T
    dL/ds_{t-1} = dL/da_t @ Ur^T             (at t = 0 the carry reads s_init, which is DETACHED, so the
                                              gradient stops there by design — but its Ur term is still
                                              accumulated: s_init is not zero in a seeded carry)
    """
    hp3, Wc, Ur, b = parents[0], parents[1], parents[2], parents[3]
    states, hps = saved["states"], saved["hp3"]
    xp = backend.xp
    gWc = xp.zeros_like(Wc.data)
    gUr = xp.zeros_like(Ur.data)
    gb = xp.zeros_like(b.data)
    ghp = xp.zeros_like(hps) if hp3.requires_grad else None
    g = grad
    for t in range(len(states) - 1, -1, -1):
        s_t = states[t]
        da = g * (1.0 - s_t * s_t)
        gWc = gWc + hps[:, t, :].T @ da
        # THE CARRY TERM AT t = 0 USES s_init, and forgetting it is a bug this pre-flight caught: with
        # W = 1 the old line `if t > 0` skipped the only step there is, so dL/dUr read exactly 0 while the
        # finite difference read a nonzero number (P111f FAIL, attempt 3) and a wired-in lever would have
        # trained Ur against a zero gradient forever.
        s_prev_t = states[t - 1] if t > 0 else saved.get("s_start")
        if s_prev_t is not None:
            gUr = gUr + s_prev_t.T @ da
        gb = gb + da.sum(axis=0)
        if ghp is not None:
            ghp[:, t, :] = da @ Wc.data.T
        g = da @ Ur.data.T
    T._accumulate(Wc, gWc)
    T._accumulate(Ur, gUr)
    T._accumulate(b, gb)
    if ghp is not None:
        T._accumulate(hp3, ghp)


T.OPS["sstate_scan"] = _bw_sstate_scan


def carry_scan(state, hp_rows, n, window):
    """Convenience: (N*W, K) per-token concepts -> (N, ds) terminal state for the answer layer."""
    hp3 = T.reshape(hp_rows, (int(n), int(window), state.K))
    return sstate_scan(hp3, state.Wc, state.Ur, state.b)

