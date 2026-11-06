"""lens/sentstate_wire.py -- graft the RUNNING SENTENCE STATE onto LensNet at construction time.

WHY A GRAFT AND NOT AN EDIT TO lens/model.py (declared, and checked by P114a)
-----------------------------------------------------------------------------
The 2026-10-26 17:2x note asks for the lever "behind sentstate=False" in lens/model.py. That edit was not made
because the trainer is under a LIVE ARM (agree_sgd04_s2, started 16:38 by the watchdog) and this project does
not edit the trainer or the model under a live arm -- the rule that kept P98 honest, stated three times in
QUEUE.txt and STATE.md. So the lever lives in a NEW module and is installed by a NEW launcher:

    lens/scripts/train_chat_sent.py   (install(); then runpy the real trainer, unmodified)

WHAT THE GRAFT DOES, EXACTLY
----------------------------
  * It wraps ONLY two methods of lens.model.LensNet: __init__ and parameters.
  * An instance is affected ONLY when `ENABLED` is true at construction time. With the flag off, the wrapper
    calls the original methods and attaches nothing, so the parameter list, the RNG draws, the forward pass and
    the checkpoint names are exactly what they were (P114a compares a full epoch line field for field).
  * With the flag on, it attaches three things and one wiring:
        PosConcepts(D, K)   per-token named concepts   hp_t = sigmoid(E[x_t] @ Wc2)
        SentenceState(K, ds) the carry               s_t  = tanh(hp_t @ Wc + s_{t-1} @ Ur + b)
        sread: ds -> V      a PARALLEL readout of the carried state
    and the answer path becomes   logits = answer(h) + sread(s_31).
  * The per-token representation is TAPPED, not re-derived: the model's own `concept` Dense is wrapped so it
    records the (N, W*D) tensor it was handed, and the wrapped `answer` Dense reads that same tensor. Nothing
    in LensNet.forward is re-implemented, so attention, the direct path, the numeric path and the mtp head all
    keep working exactly as written -- the graft only ADDS the state term to the logits it returns.

FORM CHOSEN, AND WHAT IT COSTS IN HONESTY
-----------------------------------------
The design doc widens the answer input from K to K+ds; this graft ADDS a second readout instead. Same function
class (a linear map of the state summed into the vocabulary logits), IDENTICAL parameter count (ds*V + V), but
the base answer layer keeps its shape, so a checkpoint from this arm is NOT interchangeable with the doc's
stated form. P114f checks the shapes rather than assuming them.

COST AT THE SHIP CONFIG (D=256, K=36, ds=64, V=12,002):
    Wc2 256*36 = 9,216 | Wc 36*64 = 2,304 | Ur 64*64 = 4,096 | b 64 | sread 64*12,002 + 12,002 = 780,130
    total +795,810 on 5,908,686 = +13.47%. (lens/sentstate.py's docstring quotes +783,808 for the WIDENED-
    answer form, which does not pay the separate bias row; the difference is exactly V - ds = 11,938 and is
    printed here so the two files cannot disagree silently. P114b checks the formula on the pre-flight config
    and the arithmetic is repeated in the grade file.)
"""
import numpy as np

from lens import backend
from lens import tensor as T
from lens.model import Dense
from lens.model import LensNet
import lens.model as _model
from lens.sentstate import PosConcepts, SentenceState, sstate_scan

ENABLED = False
DS = 64
SEED = 0

_ORIG_INIT = None
_ORIG_PARAMS = None


class _TapBase:
    """Delegates everything the model's own code asks of a Dense (parameters(), .W, .b, ...) to the inner
    layer, so wrapping is invisible to anyone except the answer path that adds the state term."""

    def __init__(self, inner, owner):
        self._inner, self._owner = inner, owner

    def parameters(self):
        return self._inner.parameters()

    def __getattr__(self, item):
        # only called when normal lookup fails; '_inner' lives in __dict__ so this cannot recurse
        return getattr(object.__getattribute__(self, "_inner"), item)


class _ConceptTap(_TapBase):
    """Wraps the model's concept Dense: records the per-token representation it was handed."""

    def __call__(self, x):
        self._owner._ss_x = x
        return self._inner(x)


class _AnswerTap(_TapBase):
    """Wraps the model's answer Dense: logits = answer(h) + sread(state of the tapped tokens)."""

    def __call__(self, h):
        logits = self._inner(h)
        o = self._owner
        x = getattr(o, "_ss_x", None)
        if x is None:
            raise RuntimeError("sentstate: the answer layer was called before the concept layer, so the "
                               "per-token representation was never tapped. This graft assumes the documented "
                               "forward order (concept -> answer) and refuses rather than silently dropping "
                               "the state term.")
        n = h.data.shape[0]
        xr = T.reshape(x, (n * o.W, o.D))                 # (N*W, D) per-token representation
        if xr.data.shape[0] != n * o.W or xr.data.shape[1] != o.D:
            raise RuntimeError(f"sentstate: the tapped representation has shape {tuple(x.data.shape)}, which is "
                               f"not (N*W, D) = ({n}*{o.W}, {o.D}). The state term cannot be added to a "
                               f"differently-shaped pass; refusing rather than decoding a different model.")
        hp = o.pconcept(xr)                               # (N*W, K) per-token NAMED concepts
        hp3 = T.reshape(hp, (n, o.W, o.K))                # (N, W, K)
        s = sstate_scan(hp3, o.state.Wc, o.state.Ur, o.state.b)   # (N, ds) the carry after the window
        return T.add(logits, o.sread(s))                  # (N, V)


def _wrapped_init(self, *a, **kw):
    global ENABLED
    _ORIG_INIT(self, *a, **kw)
    if not ENABLED:
        return
    rng = np.random.default_rng(int(getattr(self, "seed", 0)) + 7919)   # own stream, base draws untouched
    self.ss_ds = int(DS)
    self.pconcept = PosConcepts(self.D, self.K, rng, name="pconcept")
    self.state = SentenceState(self.K, self.ss_ds, rng, name="sstate")
    self.sread = Dense(self.ss_ds, self.V, rng, name="sread")
    self.concept = _ConceptTap(self.concept, self)
    self.answer = _AnswerTap(self.answer, self)
    self._ss_x = None


def _wrapped_parameters(self):
    ps = _ORIG_PARAMS(self)
    if getattr(self, "pconcept", None) is not None:
        ps = ps + self.pconcept.parameters() + self.state.parameters() + self.sread.parameters()
    return ps


def install(ds=64):
    """Patch LensNet. Flag OFF by default: the graft is inert until the launcher turns it on."""
    global _ORIG_INIT, _ORIG_PARAMS, DS, ENABLED
    if _ORIG_INIT is not None:
        return False                                   # already installed, idempotent
    DS = int(ds)
    _ORIG_INIT = LensNet.__init__
    _ORIG_PARAMS = LensNet.parameters
    LensNet.__init__ = _wrapped_init
    LensNet.parameters = _wrapped_parameters
    return True


def enable(ds=64):
    global ENABLED, DS
    DS = int(ds)
    ENABLED = True


def disable():
    global ENABLED
    ENABLED = False


def ship_config_cost(D=256, K=36, ds=64, V=12002):
    """The arithmetic P114b checks, in one place so no file has to hand-copy it."""
    parts = {"Wc2": D * K, "Wc": K * ds, "Ur": ds * ds, "b": ds, "sread.W": ds * V, "sread.b": V}
    return parts, sum(parts.values())


# ======================================================================================
# THE SHIPPED DECODER'S PATH  (P115a-d, predictions in lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt)
# ======================================================================================
# TWO DIFFERENT DECODE PATHS EXIST IN THIS PROJECT AND ONLY ONE OF THEM CALLS THE MODEL:
#   * NO-attention ships (every ship candidate here) -> lens/decode.py::make_cache returns None and the
#     decoder calls net.forward_arrays, which runs the graft normally. P115a measures that.
#   * ATTENTION ships -> lens/kvcache.py::WindowCache, and cached_logits_h() reads the answer matrix OFF THE
#     CACHE (cache.aW / cache.ab). It never calls net.concept/net.answer, so a grafted net decoded through it
#     returns logits with the STATE TERM MISSING, silently. That is the APILOAD-DIRECT class again.
# The fix is NOT to make the cache call the network (that would throw away the cache). It is to compute the
# term the cache cannot see from the ONE thing the cache already holds: `cache.outputs()` is exactly the
# per-token representation the graft taps in LensNet.forward (it is what the cache multiplies by cW).
# The carry is seeded at ZERO, which is what training does: the trainer draws INDEPENDENT windows, so the
# state a window trains against is that window's own carry (lens/sentstate.py says the same).
def is_grafted(net):
    """True when this network carries the sentstate graft (its six tensors are on the parameter list)."""
    return getattr(net, "sread", None) is not None and getattr(net, "pconcept", None) is not None


def _np(p):
    return np.asarray(p.data, dtype=np.float32)


def _sigmoid_np(z):
    """sigmoid exactly as the engine writes it (clip the exponent to +-60, then 1/(1+exp))."""
    return 1.0 / (1.0 + np.exp(np.clip(-z, -60.0, 60.0)))


def state_term_np(net, x_window):
    """sread(carry over the window) for ONE window, in numpy, from the per-token representation.

    x_window: (W, D) float32 -- the same tensor the graft taps. Returns (V,) float32.
    """
    ps = {p.name: _np(p) for p in net.parameters()}
    W, D, ds = int(net.W), int(net.D), int(net.ss_ds)
    x = np.asarray(x_window, dtype=np.float32).reshape(W, D)
    hp = _sigmoid_np(x @ ps["pconcept.Wc2"])                 # (W, K)
    Wc, Ur, b = ps["sstate.Wc"], ps["sstate.Ur"], ps["sstate.b"]
    s = np.zeros((ds,), dtype=np.float32)
    for t in range(W):
        s = np.tanh(hp[t] @ Wc + s @ Ur + b)                 # (ds,)
    return (s @ ps["sread.W"] + ps["sread.b"]).astype(np.float32)


def cached_logits_grafted(net, cache):
    """The KV-cached next-token logits of a GRAFTED net: the cache's answer path PLUS the state term.

    The base term comes from lens/kvcache.py::cached_logits (unchanged, the pre-edit implementation), so the
    only new arithmetic is the state term. Returns (z, h) like cached_logits_h: z (V,) float64, h (K,).
    """
    from lens.kvcache import cached_logits_h
    z, h = cached_logits_h(net, cache)
    x = np.asarray(cache.outputs(), dtype=np.float32).reshape(int(net.W), int(net.D))
    return np.asarray(z, dtype=np.float64) + state_term_np(net, x).astype(np.float64), h



# ======================================================================================
# LOADING A GRAFTED CHECKPOINT -- the silent-wrong-model guard
# ======================================================================================
# MEASURED BEFORE THIS WAS WRITTEN (lens/api.py:252-255): api.load() rebuilds the net and then iterates the
# NETWORK's parameters, so a checkpoint carrying pconcept.Wc2 / sstate.Wc / sstate.Ur / sstate.b / sread.W /
# sread.b loads SILENTLY into an un-grafted model: the six tensors are ignored, no KeyError is raised, and
# every post-hoc consumer (the readability gate, the talk decider, the exporters, the server) then reports
# numbers for a DIFFERENT model than the one that was trained. That is the same defect class as
# APILOAD-DIRECT (the tied path silently dropped on load), which cost this project a full re-measurement.
# So: a grafted checkpoint is loaded only through load_grafted(), which enables the graft FIRST, reads ds from
# the checkpoint itself, and then REFUSES if the file's parameter set and the model's parameter set differ at
# all -- in either direction.
GRAFT_PARAMS = ("pconcept.Wc2", "sstate.Wc", "sstate.Ur", "sstate.b", "sread.W", "sread.b")


def load_grafted(path, device="cpu"):
    import json as _json
    from lens.api import load as _api_load
    z = np.load(path, allow_pickle=False)
    keys = set(z.files) - {"__meta__"}
    grafted = [k for k in GRAFT_PARAMS if k in keys]
    if not grafted:
        raise KeyError(f"{path} carries no sentstate tensors: it is a base checkpoint, so load it with "
                       f"lens.api.load() as usual. Refusing to guess which form was meant.")
    if len(grafted) != len(GRAFT_PARAMS):
        raise KeyError(f"{path} carries a PARTIAL graft {sorted(grafted)} of {len(GRAFT_PARAMS)} tensors. "
                       f"A partial graft would load a state path with uninitialised pieces; refusing.")
    ds = int(z["sread.W"].shape[0])
    install(ds=ds)
    enable(ds=ds)
    try:
        net = _api_load(path, device=device)
    finally:
        disable()                                     # leave the module's global state as we found it
    names = {p.name for p in net.parameters()}
    if names != keys:
        raise KeyError("the checkpoint and the rebuilt model disagree on their parameters, so this load would "
                       "have been silently partial:\n  only in file : "
                       f"{sorted(keys - names)}\n  only in model: {sorted(names - keys)}")
    return net
