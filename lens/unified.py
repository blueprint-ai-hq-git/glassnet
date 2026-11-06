"""lens/unified.py - ONE training method instead of SFT then RL then distillation in stages.

Yahya (2026-10-20): "make one single training method for distillation and RL, there is no SFT or any other
training method because those two are more than enough ... in RL we can put our own dataset and give it to
learn and do it by itself and solve its problems, distillation we can just put a distilled dataset that would
be just normal training, or with a teacher model ... so people do not have to go in stages, pretrain then
train then RL. There is one training method."

So there is exactly one loop. What changes is only WHERE THE TARGET COMES FROM:
    target from a VERIFIER   -> reinforcement: the model answers, the verifier says right or wrong, it is
                                reinforced on what it got right. Give it a dataset of problems and it solves
                                them by itself.
    target from a TEACHER    -> distillation: the model matches a teacher's answer (soft or hard).
    target from the DATA     -> ordinary training on a distilled dataset.
Same loop, same gates, same readable output. The mode is a parameter, not a different pipeline.

THE RULE THAT MAKES IT LEARN INSTEAD OF MEMORISE (from LEARN-NOT-MEMORISE.md), applied inside the loop:
    gap gate       : a step that widens train-minus-heldout is REVERTED
    agreement gate : a step whose per-example gradients disagree is scaled down or refused
Both numbers are printed every step, so the run is readable while it trains.
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _loss_and_grads(X, y, w, b):
    """Tiny linear readout, our own maths: squared error and its gradient. Kept deliberately small so the
    unified loop can be tested for real instead of described."""
    pred = X @ w + b
    err = pred - y
    loss = float(np.mean(err ** 2))
    g = (2.0 / len(X)) * (X.T @ err)
    gb = float((2.0 / len(X)) * err.sum())
    return loss, g, gb, pred


def _per_example_grads(X, y, w, b):
    """Each example's own gradient direction, for the agreement gate."""
    pred = X @ w + b
    err = (pred - y)[:, None]
    return 2.0 * err * X


class GapGate:
    """THE memorisation gate, in one place so it can be calibrated instead of asserted.

    It keeps an exponential moving average (EMA) of the training loss and of the held-out loss, because a
    single mini-batch is far too noisy to call memorisation on: the first version of this gate compared one
    batch and refused 202 of 300 honest steps.

    variant 'ema_band'   : refuse when the smoothed held-out loss is more than tol above its own previous
                           value. This was the shipped rule. PROBLEM (see lens/scripts/gap_gate_calibration.py):
                           with beta=0.9, cand = 0.9*old + 0.1*new, so cand > old*(1+tol) needs
                           new > old*(1+10*tol) - a 50% jump at tol=0.05. The smoothing and the tolerance
                           multiply, so it barely ever fires.
    variant 'divergence' : the rule as WRITTEN in LEARN-NOT-MEMORISE.md section 2a - did held-out loss get
                           worse while training loss improved? Two smoothed series, compared with each other
                           instead of one series against itself. This is the one the calibration measures.
    variant 'gap_level'  : DETECTOR 3a (added 2026-10-20, after detectors 1 and 2 both failed their own
                           calibration). Refuse while the training loss sits persistently BELOW the held-out
                           loss by a relative margin: train < held * (1 - tol). That is the memorisation
                           signature itself (fitting the batch better than fresh data), and it is the gap the
                           doc names, as a LEVEL. On a learnable corpus both losses fall together and the ratio
                           stays near 1, so it stays quiet; on a corpus that can only be memorised the training
                           loss goes to 0 while the held-out loss cannot, so the ratio goes to 0.
    variant 'rate_ratio' : DETECTOR 3b. Both losses are EMA-smoothed, then their RELATIVE improvements are
                           compared: refuse when the train loss improved by more than 'min_rate' this step
                           while the held-out loss improved less than 'ratio' times as fast. Scale-free, and it
                           goes quiet when BOTH series have stopped moving - which is exactly the converged
                           tail where detectors 1 and 2 fired wrongly on the learnable corpus.
    variant 'trend_band' : DETECTOR 4 (added after detectors 1, 2 and 3 ALL failed their own calibration
                           online). The EMA of the held-out loss compared with ITS OWN VALUE 'lag' steps ago:
                           refuse when cand_h > ema_h[now-lag] * (1 + tol). At lag=1 this is detector 1 exactly,
                           so the lag is the single thing that changes. Why: every failed detector compares a
                           SINGLE STEP against another smoothed value, and a single step of a noisy series
                           carries no information about which of the two things is happening. The trend over a
                           window does: the learnable corpus's held-out loss falls for the whole run, the
                           memorisable one rises.
    variant 'gap_trend'  : DETECTOR 4b, the same lagged comparison on the GAP (train - held-out): refuse when
                           the gap has widened (gone more negative) by more than tol times the held-out level
                           over the lag.
    Everything about a step (both EMAs, both losses, the gap, the verdict) is a named printable number.
    """

    def __init__(self, beta=0.9, tol=0.01, variant="divergence", ratio=0.5, min_rate=0.002, lag=25):
        assert variant in ("ema_band", "divergence", "best_band", "gap_level", "rate_ratio",
                           "trend_band", "gap_trend", "off")
        self.beta = float(beta)
        self.tol = float(tol)
        self.variant = variant
        self.ratio = float(ratio)          # rate_ratio: how much slower than train the held-out may improve
        self.min_rate = float(min_rate)    # rate_ratio: ignore steps where the train loss barely moved
        self.lag = int(lag)                # trend_band / gap_trend: how far back the trend is measured
        self.ema_train = None
        self.ema_held = None
        self.best_held = None
        self.hist_t = []                   # committed EMAs, for the lagged trend detectors
        self.hist_h = []
        self.checked = 0
        self.refused = 0
        self.gap_first = None
        self.gap_last = None

    def evaluate(self, train_loss, held_loss):
        """Look at the CANDIDATE step's losses. Returns (refuse, cand_train_ema, cand_held_ema, gap)."""
        self.checked += 1
        b = self.beta
        cand_t = train_loss if self.ema_train is None else b * self.ema_train + (1 - b) * train_loss
        cand_h = held_loss if self.ema_held is None else b * self.ema_held + (1 - b) * held_loss
        gap = cand_t - cand_h
        if self.ema_train is None or self.ema_held is None:
            refuse = False                      # nothing to compare against yet
        else:
            refuse = self.decide(cand_t, cand_h)
        if refuse:
            self.refused += 1
        return refuse, cand_t, cand_h, gap

    def decide(self, cand_t, cand_h):
        """THE rule, in one place, so the offline screen in lens/scripts/gap_gate_detector3.py judges the SAME
        decision function that ships. Every variant sees only the two smoothed losses."""
        if self.variant == "off":
            return False                        # control: the gate switched off entirely
        if self.variant == "ema_band":
            return bool(cand_h > self.ema_held * (1 + self.tol))
        if self.variant == "best_band":
            # DETECTOR 2 (added 2026-10-20 after detector 1 failed its calibration, see PART 4 of
            # lens/scripts/gap_gate_calibration.py): memorisation is held-out loss rising above the BEST it
            # has ever been, not above its own drifting average. On a corpus that can be learned the held-out
            # loss falls, so it sits at its best almost all the time and the gate stays quiet; on a corpus that
            # can only be memorised it climbs away from its best immediately.
            # MEASURED: also FAILS its calibration (part 5) - on the learnable corpus the held-out loss stops
            # falling once it reaches its own noise floor, and then this fires about half the time.
            return bool(self.best_held is not None and cand_h > self.best_held * (1 + self.tol))
        if self.variant == "gap_level":
            # DETECTOR 3a: the training loss persistently BELOW the held-out loss by a relative margin.
            return bool(cand_t < cand_h * (1.0 - self.tol))
        if self.variant == "rate_ratio":
            # DETECTOR 3b: the train loss is improving (more than min_rate) while the held-out loss is not
            # improving at a comparable relative rate.
            d_t = (self.ema_train - cand_t) / max(abs(self.ema_train), 1e-12)
            d_h = (self.ema_held - cand_h) / max(abs(self.ema_held), 1e-12)
            return bool(d_t > self.min_rate and d_h < self.ratio * d_t)
        if self.variant == "trend_band":
            # DETECTOR 4: the smoothed held-out loss against its OWN value 'lag' steps ago. lag=1 is exactly
            # detector 1; the lag is the only difference.
            if len(self.hist_h) < self.lag:
                return False                    # not enough history yet
            ref = self.hist_h[-self.lag]
            return bool(cand_h > ref * (1.0 + self.tol))
        if self.variant == "gap_trend":
            # DETECTOR 4b: the gap (train - held) against its own value 'lag' steps ago: has the gap widened
            # (gone more negative) by more than tol times the held-out level?
            if len(self.hist_h) < self.lag:
                return False
            ref_h = self.hist_h[-self.lag]
            ref_gap = self.hist_t[-self.lag] - ref_h
            return bool((cand_t - cand_h) < ref_gap - self.tol * max(abs(ref_h), 1e-12))
        # divergence: held-out worse WHILE train improves (the wording of LEARN-NOT-MEMORISE.md section 2a)
        return bool(cand_h > self.ema_held * (1 + self.tol) and cand_t <= self.ema_train)

    def commit(self, cand_t, cand_h):
        """Apply the step's EMAs. Only accepted steps move the state."""
        self.ema_train = cand_t
        self.ema_held = cand_h
        self.best_held = cand_h if self.best_held is None else min(self.best_held, cand_h)
        self.hist_t.append(cand_t)
        self.hist_h.append(cand_h)
        if self.gap_first is None:
            self.gap_first = cand_t - cand_h
        self.gap_last = cand_t - cand_h


class AgreementGate:
    """THE SECOND DETECTOR of LEARN-NOT-MEMORISE.md, section 2b: does the batch AGREE on a direction?

    The doc's wording: "per-example gradients are compared. Learning produces a direction the examples AGREE
    on; memorising produces a different direction per example. Low agreement -> the step is scaled down or
    refused. Both numbers are printed every step."

    THE NUMBER (one, named, printable): the batch is cut into G equal sub-batches, each sub-batch gets its own
    forward+backward, and

        A = || sum_g grad_g ||  /  sum_g || grad_g ||

    over every parameter of the model concatenated. A = 1 when every sub-batch points the same way, A ~
    1/sqrt(G) when the sub-batch directions are mutually independent (0.3536 at G=8), A = 0 when they cancel.
    It is the coherence of the batch gradient, and it is exactly the quantity the doc asks to be printed.

    WHY IT IS MEASURED ON SUB-BATCHES RATHER THAN ONE EXAMPLE AT A TIME: the library's loss is a MEAN over
    the samples it is handed and its backward accumulates into p.grad, so one example at a time is the same
    measurement with a noisier estimate and G times the Python overhead. G sub-batches of the SAME batch
    answer the same question about the same gradient. Stated, not hidden: a sub-batch gradient is the mean
    over its own samples, so A is measured on the direction each GROUP of samples agrees on, not each sample.

    VARIANTS (each one is a decision function over the two numbers it is given, nothing else):
      'off'       control: refuses nothing. Must be digit-identical to no gate at all.
      'level'     refuse when A_train < tol. A FIXED threshold, so it is only as good as the human who picked
                  it - it is here to be measured, not because it is believed.
      'ema_band'  refuse when A_train < ema_A - tol (beta 0.5). The direct analogue of the gap gate's shipped
                  detector 1. Its known weakness is the same one: one noisy number compared with another.
      'vs_probe'  refuse when A_train < A_probe - tol. The item's own mechanism, SELF-CALIBRATED: the model
                  compares how much the data it is training on agrees with how much data it has NEVER seen
                  agrees, so there is no human-chosen scale in it. If the batch pulls in its own directions
                  harder than fresh data does, the batch is being fitted, not learned from.

    DEFAULTS ARE THE METHOD'S OWN: variant 'vs_probe', tol 0.0, beta 0.5, groups 8. The first step can never
    be refused (there is nothing to compare against yet) - stated, because it is a real property of the rule.
    """

    def __init__(self, variant="vs_probe", tol=0.0, beta=0.5, groups=8):
        assert variant in ("off", "level", "ema_band", "vs_probe")
        self.variant = variant
        self.tol = float(tol)
        self.beta = float(beta)
        self.groups = int(groups)
        self.ema = None
        self.checked = 0
        self.refused = 0
        self.a_first = None
        self.a_last = None

    def evaluate(self, a_train, a_probe=None):
        """Look at this step's coherence numbers. Returns (refuse, cand_ema)."""
        self.checked += 1
        if self.a_first is None:
            self.a_first = float(a_train)
        self.a_last = float(a_train)
        b = self.beta
        cand = float(a_train) if self.ema is None else b * self.ema + (1.0 - b) * float(a_train)
        refuse = self.decide(a_train, a_probe)
        if refuse:
            self.refused += 1
        return refuse, cand

    def decide(self, a_train, a_probe):
        """THE rule, in one place, so a reader judges the same decision function that ships."""
        if self.variant == "off":
            return False                        # control
        if self.variant == "level":
            return bool(a_train < self.tol)
        if self.variant == "ema_band":
            return bool(self.ema is not None and a_train < self.ema - self.tol)
        if self.variant == "vs_probe":
            # the first step has no reference: a rule that refuses on no evidence is a coin toss
            return bool(a_probe is not None and a_train < a_probe - self.tol)
        return False

    def commit(self, cand):
        """Apply the step's EMA. Only accepted steps move the state (same as the gap gate)."""
        self.ema = float(cand)


def binary_sign_ladder(verifier, pred, y_true, tol=0.5, dmax=40.0, spacing=1.5):
    """A DIRECTION out of a RIGHT/WRONG verdict, with no rank oracle (PLAN item verifier3, 2026-10-21).

    THE PROBLEM IT SOLVES (measured, lens/runs/UNIFIED-VERIFIER.txt PART 6): a verifier that only says right
    or wrong stops giving a direction once its own tolerance is satisfied, so the shipped fallback parks
    every example on the edge of the band |pred - y| < tol and the model ends up with a slope error of
    6.797% at EVERY probe step tried - 97% exact on the pairs it trained on and 0.0% on pairs it never saw.
    Shrinking the probe step cannot fix it; the dead zone is the VERDICT's, not the probe's.

    THE FIX: the verdict is a membership test. A candidate pred + d is "right" iff |pred + d - y| < tol,
    i.e. iff |pred - y| lies in the window (d - tol, d + tol) on that side. So ask about pred + d AND
    pred - d for one rung d: since d >= tol the two windows cannot both be satisfied, and EXACTLY ONE of
    them is right exactly when |pred - y| is in (d - tol, d + tol) - which names the SIDE of the target.
    That is a direction, and it costs two verdicts.
    Scan a LADDER of rungs d = tol, tol + spacing*tol, ... with spacing < 2 (so consecutive windows
    OVERLAP and no residual can fall between two rungs) and take the FIRST rung that resolves. The
    resolution of the rule is the rung spacing, NOT the verifier's tolerance, and NOT the probe step.
    Every example keeps its own direction, so there is no subset bias, and an example the model already
    answers correctly still gets its direction (it is not parked, which is what created the slope error).

    tol is the verifier's DECLARED tolerance (the 0.5 in `|round(pred) - y| < 0.5`), spacing is in units of
    tol, dmax bounds the ladder (rungs above it are not asked).

    Returns (sign, questions, unresolved): sign is +1 where the target is ABOVE pred, -1 where it is below,
    0 where the verdicts carry no direction (both sides wrong: |pred - y| sits exactly on a window edge, or
    is beyond the top rung); questions counts the verifier calls made; unresolved counts the 0s.
    """
    n = len(pred)
    sign = np.zeros(n, dtype=np.float64)
    found = np.zeros(n, dtype=bool)
    questions = 0
    d = float(tol)
    step = float(spacing) * float(tol)
    while d <= float(dmax) + float(tol) and not found.all():
        up = np.asarray(verifier(pred + d, y_true)).astype(bool)
        dn = np.asarray(verifier(pred - d, y_true)).astype(bool)
        if up.shape != (n,):
            up = np.broadcast_to(up, (n,))
        if dn.shape != (n,):
            dn = np.broadcast_to(dn, (n,))
        questions += 2
        only_up = up & ~dn & ~found
        only_dn = dn & ~up & ~found
        sign[only_up] = 1.0
        sign[only_dn] = -1.0
        found |= only_up | only_dn
        d += step
    return sign, questions, int((~found).sum())


def train_unified(X, y, *, mode="verifier", teacher=None, verifier=None, heldout_frac=0.25,
                  steps=300, lr=0.05, seed=0, agreement_min=0.02, verbose=True,
                  gate=None, gate_beta=0.5, gate_tol=0.0, agreement_mode="one", trace=False,
                  verifier_mode="probe3", verifier_rank=None, probe_step=1.0 / 16.0, diag=None,
                  probe_schedule=None, probe_normalise=True,
                  binary_probe="ladder", ladder_tol=0.5, ladder_max=40.0, ladder_spacing=1.5,
                  ladder_flip=False):
    """THE one training method. mode is 'verifier', 'teacher' or 'data'.

    verifier_mode decides HOW the verifier is asked. DEFAULT CHANGED 2026-10-21 18:3x (PLAN item verifier2):
    the default is now 'probe3' with probe_step = 1/16 and probe_normalise = True, because 'hard_quarter' was
    MEASURED to collapse (9.0% exact on trained pairs, 0.0% on unseen, lens/runs/UNIFIED-RUN-pre-probe3.txt)
    while the probe3 rule reaches the true slope and 100% exact on pairs it never saw
    (lens/runs/UNIFIED-VERIFIER.txt, PART 4, P54a/b/c PASS). Predictions written before the change:
    lens/runs/UNIFIED-VERIFIER-PREDICTIONS-ADDENDUM4.txt (P55a-P55k).
      'probe3'       - THE DEFAULT. The verifier is a COMPARISON oracle: it names which of three candidates
                       (pred, pred + probe_step, pred - probe_step) is closest to being right, and the model
                       is trained toward the winner. It is the same question a right/wrong verdict answers,
                       asked in the form that carries a DIRECTION: right/wrong is the limit of this as the
                       probe step goes to infinity. With probe_normalise=True the update is divided by the
                       probe step, so the step is a normalised sign-descent on the residual (median
                       regression) whose solution on noiseless data is the true weights. Every example is
                       kept, so no subset bias is possible. Needs verifier_rank(cands, y_true) -> index per
                       example; with no rank oracle the DIRECTION comes from binary_probe (below).
                       MEASURED, 12000 steps, 3 seeds: slope error 0.47/0.45/0.44% (bar <= 1%) and 100.0%
                       exact on trained AND on unseen pairs on 3 of 3 (P54a/b/c, P55g/h).
    binary_probe decides where the sign comes from when there is NO rank oracle. DEFAULT CHANGED 2026-10-21
    19:3x (PLAN item verifier3) to 'ladder', because the ladder was MEASURED to carry the rule with a plain
    right/wrong verifier: 100.0% exact on trained AND unseen pairs, slope error 0.19/0.23/0.19%, 3 of 3
    seeds (lens/runs/UNIFIED-LADDER.txt, PART 7b arm L2) against the old fallback's 97/98/99% trained and
    **0.0% unseen** at slope error 6.80/6.49/5.95%. Predictions written first:
    lens/runs/UNIFIED-LADDER-PREDICTIONS.txt (P57a-P57l).
        'ladder'  - THE DEFAULT NOW. A direction OUT OF THE RIGHT/WRONG VERDICT ALONE: ask about pred +- d
                    on a ladder of rungs spaced 1.5*tol (tol = ladder_tol, the verifier's declared
                    tolerance) and take the first rung where exactly one side is "right"; that side names
                    where the target is. Rung windows overlap, so no residual inside the ladder's range
                    falls between two rungs, and the rule's dead zone is a single point (the residual
                    exactly 0) instead of the verifier's whole 0.5 band. Cost is the verifier questions,
                    logged per step in log['ladder_q_per_example']: mean 3.9-4.8 calls per step over a
                    12000-step run, 2.0 per step once converged (one rung), against the old fallback's 3.
                    Residuals beyond ladder_max + tol are NOT resolved (they park, like the old rule).
        'nearest' - the FORMER behaviour, KEPT AS A NAMED VARIANT so every stored number that describes it
                    is still reproducible on request: the binary verifier is asked about the three
                    candidates and the nearest one wins. MEASURED (lens/runs/UNIFIED-VERIFIER.txt PART 5/6):
                    97/98/99% exact on trained pairs, 0.0% on unseen, slope error 6.797% at EVERY probe
                    step - it parks each example on the edge of the verifier's own 0.5 band.
      'hard_quarter' - THE FORMER DEFAULT, KEPT AS A NAMED VARIANT so its numbers stay reproducible on
                       request (arm A of lens/scripts/unified_verifier_test.py names it explicitly). The
                       verifier says right or wrong per example; when nothing is right the closest 25% of
                       the batch by |pred - y| is kept and the step is built from those. MEASURED FAILURE:
                       at initialisation pred ~ 0, so "closest" means "smallest target", the subset is
                       biased LOW, the model is pulled to the low end of the range and the frontier never
                       crosses the upper half - 9.0% exact on trained pairs, 0.0% on unseen, and at 12000
                       steps it is WORSE still (slope error 74-85%). It is NOT the default any more.
    diag, if a list is passed, collects one dict per verifier step (selected/batch target means, etc.) so the
    mechanism can be measured instead of asserted. It changes no arithmetic.

    agreement_mode decides what the agreement number does to a step that is NOT refused (see section 3 of
    lens/scripts/unified_harness_probe.py for the measurement behind the default):
        'floor' - step scaled by max(agreement, 0.05). The former default: MEASURED WRONG, see below.
        'raw'   - step scaled by agreement, no floor.
        'one'   - step not scaled at all; agreement is then only a refusal test. THIS IS THE DEFAULT.

    WHY 'one' (measured, 3 corpus seeds x 3 training seeds, lens/scripts/unified_harness_probe.py section 3):
    on the pairs harness, with the gap gate on, 'floor' and 'raw' reach 0.0% exact on pairs the model never
    saw while 'one' reaches 100.0%; on the learnable corpus 'one' ends at the lowest held-out loss
    (0.00237+-0.00046 vs 0.00254+-0.00046) and on the memorisable corpus at the LOWEST damage as well
    (1.0152+-0.1554 vs 1.0352+-0.1664). The throttle only ever slowed the honest steps: the gap gate is what
    refuses the memorising ones. The agreement NUMBER is still computed, printed and used as a refusal test.
    """
    assert mode in ("verifier", "teacher", "data")
    assert binary_probe in ("nearest", "ladder"), binary_probe
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_held = max(8, int(len(X) * heldout_frac))
    hi, ti = order[:n_held], order[n_held:]
    Xh, yh = X[hi], y[hi]            # the model's own probe: never trained on
    Xt, yt = X[ti], y[ti]
    # BUG FOUND 2026-10-20 by lens/scripts/distillation_test.py, and fixed here: in teacher mode the probe
    # was still scored against the LABELS. The model was trained to match the teacher while the gap gate
    # watched a target it was never asked to match, so held-out loss could not fall and the gate refused the
    # steps that were doing the distilling. Measured before the fix, teacher = a+b+3, 3 seeds, 3000 steps:
    # held-out matched the teacher 6.7% and the labels 14.7% - it learned NEITHER. The probe must be scored
    # against the same target the model is being trained on; that is the whole point of "where the target
    # comes from" being the one thing the modes differ in.
    if mode == "teacher" and teacher is not None:
        yh = teacher(Xh)
        yt = teacher(Xt)

    d = X.shape[1]
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    # THE MEMORISATION GATE (2026-10-20). Two EMAs, one rule, calibrated in
    # lens/scripts/gap_gate_calibration.py on a corpus that CAN be memorised and one that CAN be learned:
    # the gate must fire on the first and must not fire on the second, or it is decoration.
    # MEASURED, 3 corpus seeds x 3 training seeds (lens/runs/gap_gate_calibration.json, parts 1-5):
    #   the previously shipped rule (ema_band, beta=0.9, tol=0.05) refused 0/300 steps on the memorisable
    #   corpus   FAIL - it was decoration.
    #   ema_band beta=0.5 tol=0.0, with the step NOT throttled (the shipped default below): it fires on 83.7%
    #   of the memorisable corpus and refuses 66.0% of the learnable corpus. Against the cost bar declared in
    #   part 4 (RULE refusals <= 20%) that is a FAIL. The OUTCOME test (part 2, same seeds, gate on vs off)
    #   PASSES: memorisable corpus held-out 1.0152+-0.1554 with the gate vs 1.2637+-0.2562 without it, and the
    #   learnable corpus 0.0024+-0.0005 vs 0.0028+-0.0005, i.e. refusing does prevent the damage without
    #   costing held-out. REVIEW: this is the one place the ship decision disagrees with a pre-declared bar
    #   (see AUTOPILOT-LOG.md, 2026-10-20), because the <=20% bar is a cost proxy and the outcome test is the
    #   stronger measurement; the memorisation rule is kept ON rather than silently switched off.
    #   Parts 4 and 5 also tested the interaction with the step size and a second detector ('best_band':
    #   held-out against its own best instead of its own average). Neither produced an admissible
    #   configuration at the shipped step size, so detector 1 stands with its false-fire cost recorded.
    #   2026-10-20 14:5x - DETECTORS 3 AND 4 ALSO FAILED, AND THE GATE IS NO LONGER DECORATION. Four more
    #   variants were added to GapGate.decide and judged ONLINE (the offline replay was measured unfaithful:
    #   NOISE 0.943 / RULE 0.739 replayed against the known online 0.837 / 0.660, deviation 0.185):
    #     'gap_level' (train below held)      FAIL, and at tol 0.1/0.2 it DESTROYS learning - the learnable
    #                                         corpus ends at 1203.9x its gate-off held-out loss (3.31 vs 0.00275)
    #     'rate_ratio' (relative improvement) FAIL, fires on both corpora (NOISE 0.85-0.96, RULE 0.45-0.85)
    #     'trend_band' / 'gap_trend' (lagged) 0 of 40 pass; best 'trend_band' beta=0.9 lag=50 tol=0.02 fires on
    #                                         the memorisable corpus 0.787/0.833/0.277 per seed and refuses only
    #                                         0.037/0.083/0.187 of the learnable one (RULE held-out 0.987x
    #                                         gate-off, NOISE 0.990x) - it misses the NOISE >= 0.50 bar on seed 2,
    #                                         so it is RECOMMENDED TO THE USER, NOT APPLIED. Default unchanged.
    #   WHAT IS NOW MEASURED: the gate DOES fire on a corpus that can be memorised (LEARN-NOT-MEMORISE.md item 2
    #   is met) but nothing fires there AND stays off the learnable corpus on every seed, because once the
    #   learnable corpus reaches its noise floor its held-out loss is a noisy flat series. Files:
    #   lens/runs/GAP-GATE-DETECTOR3.txt, lens/runs/GAP-GATE-DETECTOR4.txt (+ .json, predictions first).
    if gate is None:
        gate = GapGate(beta=gate_beta, tol=gate_tol, variant="ema_band")
    log = {"mode": mode, "steps": steps, "refused_gap": 0, "refused_agreement": 0, "accepted": 0,
           "ema_held_start": None, "ema_held_end": None,
           "gap_first": None, "gap_last": None, "heldout_first": None, "heldout_last": None,
           "train_last": None, "trace": [] if trace else None,
           "ladder_questions": 0, "ladder_unresolved": 0, "ladder_examples": 0,
           "ladder_q_per_example": []}

    for step in range(steps):
        idx = rng.integers(0, len(Xt), size=min(16, len(Xt)))
        Xb, yb = Xt[idx], yt[idx]

        # ---- WHERE THE TARGET COMES FROM (the only thing that differs between the three modes)
        grad_scale = 1.0        # probe3 with normalise=True needs this: the step's magnitude is proportional
                                # to the probe step s, so a small s would stall the run (measured: arm E)
        if mode == "verifier":
            _, _, _, pred = _loss_and_grads(Xb, yb, w, b)
            y_true = yb
            if verifier_mode == "probe3":
                # THE FIX (2026-10-21). The verifier is a comparison oracle: it names which of three
                # candidates is closest to being right, and the model is trained toward the winner. The
                # candidate set contains pred itself, so an example the model already answers correctly is
                # trained toward its own answer - a no-op, not damage. Every example is kept, so there is no
                # subset of the target distribution to be biased toward, which is what killed the shipped rule.
                s_now = probe_step if probe_schedule is None else float(probe_schedule(step))
                cands = np.stack([pred, pred + s_now, pred - s_now], axis=1)
                if verifier_rank is not None:
                    pick = np.asarray(verifier_rank(cands, y_true), dtype=int)
                elif binary_probe == "ladder":
                    # PLAN item (verifier3), 2026-10-21: a DIRECTION out of the right/wrong verdict alone.
                    # Ask about pred +- d on a ladder of rungs and take the first rung where exactly one
                    # side is "right"; that side names where the target is. The dead zone is then the rung
                    # spacing, not the verifier's tolerance and not the probe step. Predictions first:
                    # lens/runs/UNIFIED-LADDER-PREDICTIONS.txt (P57a-P57l).
                    if verifier is None:
                        sg, nq, n_unres = np.zeros(len(pred)), 0, int(len(pred))
                    else:
                        sg, nq, n_unres = binary_sign_ladder(verifier, pred, y_true, tol=ladder_tol,
                                                             dmax=ladder_max, spacing=ladder_spacing)
                    if ladder_flip:
                        sg = -sg      # CONTROL ONLY (lens/scripts/unified_ladder_test.py arm L3): proves
                                      # the resolved direction is what carries the learning. Never set
                                      # this in a real run.
                    log["ladder_questions"] += nq
                    log["ladder_unresolved"] += n_unres
                    log["ladder_examples"] += int(len(pred))
                    # ONE verifier call asks EVERY example in the batch one question, so the number of
                    # calls made in this step IS the questions per example per step. (The first version of
                    # this log divided by the batch size, which understated the cost 16x - a units error,
                    # caught in the smoke test before any recorded number was taken.)
                    log["ladder_q_per_example"].append(float(nq))
                    pick = np.where(sg > 0, 1, np.where(sg < 0, 2, 0))
                else:
                    # default ranking: the verifier's own right/wrong test, then distance. This is the
                    # honest fallback when only a binary verifier exists.
                    # BUG FOUND AND FIXED 2026-10-21 18:3x by running the smoke test on the new default
                    # (BUGS.md): this branch had NEVER been executed before - every previous arm passed
                    # verifier_rank, so the fallback was dead code - and it called verifier(cands, y_true)
                    # with a (n,3) matrix against an (n,) target, which the standard right/wrong verifier
                    # `np.abs(np.round(pred) - yb) < 0.5` cannot broadcast:
                    #   ValueError: operands could not be broadcast together with shapes (16,3) (16,)
                    # The verifier is asked about each of the three candidates SEPARATELY, which is the
                    # only thing a right/wrong verifier can answer.
                    ok = np.stack([verifier(cands[:, j], y_true) for j in range(cands.shape[1])], axis=1) \
                        if verifier is not None else np.zeros_like(cands, dtype=bool)
                    any_ok = ok.any(axis=1)
                    pick = np.where(any_ok, np.argmax(ok, axis=1),
                                    np.argmin(np.abs(cands - y_true[:, None]), axis=1))
                yb = cands[np.arange(len(pred)), pick]
                if probe_normalise:
                    # ARM G (2026-10-21): the candidate sits s away from pred, so |grad| ~ s. Normalising by
                    # 1/s makes the step a pure sign-descent step, which is median regression - its solution
                    # on noiseless data is the true w, and the small s keeps the zero-gradient band
                    # |err| < s negligible instead of leaving the slope short (arm B: slope 6.5% low, 95%
                    # exact on trained pairs and 0% off them).
                    grad_scale = 1.0 / max(s_now, 1e-12)
                log["probe3_steps"] = log.get("probe3_steps", 0) + 1
                if diag is not None:
                    diag.append({"step": step, "rule": "probe3", "n": int(len(y_true)),
                                 "exact_candidate": float((np.round(cands) == y_true[:, None]).any(axis=1).mean()),
                                 "mean_y_batch": float(np.mean(y_true)),
                                 "mean_target": float(np.mean(yb)),
                                 "mean_pred": float(np.mean(pred))})
            elif verifier is not None:
                ok = verifier(pred, yb)
                if ok.sum() == 0:
                    # COLD START (found by running it, 2026-10-20): at initialisation nothing is right, so a
                    # pure right/wrong verifier gives NO signal and the loop deadlocks (0 steps accepted).
                    # Partial credit instead: keep the closest attempts and reinforce those. This is how a
                    # model can "solve problems by itself" from a standing start rather than waiting for luck.
                    # MEASURED 2026-10-21 (lens/scripts/unified_verifier_test.py): this rule is BIASED - at
                    # pred ~ 0 the closest quarter of the batch is the quarter with the SMALLEST targets, so
                    # the step is built from the low end of the range and the model is pulled down there.
                    dist = np.abs(pred - yb)
                    keep = dist <= np.quantile(dist, 0.25)
                    if diag is not None:
                        diag.append({"step": step, "rule": "hard_quarter", "n": int(len(yb)),
                                     "n_kept": int(keep.sum()),
                                     "mean_y_batch": float(np.mean(yb)),
                                     "mean_y_selected": float(np.mean(yb[keep])),
                                     "mean_pred": float(np.mean(pred))})
                    Xb, yb = Xb[keep], yb[keep]
                    log["partial_credit_steps"] = log.get("partial_credit_steps", 0) + 1
                else:
                    Xb, yb = Xb[ok], yb[ok]
        elif mode == "teacher":
            yb = teacher(Xb)                      # the teacher's answer replaces the label
        # mode == "data": yb stays as it is

        # ---- candidate step
        loss_b, g, gb, _ = _loss_and_grads(Xb, yb, w, b)
        peg = _per_example_grads(Xb, yb, w, b)
        flat = peg / (np.linalg.norm(peg, axis=1, keepdims=True) + 1e-12)
        mean_dir = flat.mean(axis=0)
        mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
        agreement = float((flat @ mean_dir).mean())   # 1 = all examples agree, 0 = they disagree completely

        if agreement < agreement_min:
            # only near-TOTAL disagreement is refused (agreement_min defaults to 0.02). Otherwise the step is
            # SCALED by agreement: examples disagreeing slows the step down without blocking it. (The first
            # version refused at 0.25 and blocked 161 of 300 legitimate learning steps -- found by running it.)
            log["refused_agreement"] += 1
            if trace:
                # the gate never sees these steps, so the offline screen must skip them too
                log["trace"].append({"step": step, "by_agreement": True, "t": None, "h": None,
                                     "accepted": False, "agreement": float(agreement)})
            if verbose and step % 50 == 0:
                print(f"  step {step:4d} REFUSED (agreement {agreement:.3f} < {agreement_min})")
            continue

        step_scale = {"floor": max(agreement, 0.05), "raw": max(agreement, 1e-3), "one": 1.0}[agreement_mode]
        w_new, b_new = (w - lr * step_scale * grad_scale * g,
                        b - lr * step_scale * grad_scale * gb)
        loss_h_before, _, _, _ = _loss_and_grads(Xh, yh, w, b)
        loss_h_after, _, _, _ = _loss_and_grads(Xh, yh, w_new, b_new)
        gap_before = loss_b - loss_h_before

        # THE RULE, on the smoothed signal: a step is refused when the held-out loss is persistently getting
        # worse while the training loss is still improving. Both smoothed numbers are printed every step.
        train_after = _loss_and_grads(Xb, yb, w_new, b_new)[0]
        refuse, cand_t, cand_h, gap_ema = gate.evaluate(train_after, loss_h_after)
        gap_after = gap_ema
        if trace:
            log["trace"].append({"step": step, "by_agreement": False, "t": float(train_after),
                                 "h": float(loss_h_after), "accepted": not refuse,
                                 "agreement": float(agreement)})
        if refuse:
            log["refused_gap"] += 1
            if verbose and step % 50 == 0:
                print(f"  step {step:4d} REFUSED (held-out EMA {gate.ema_held:.4f} -> {cand_h:.4f} "
                      f"while train EMA {gate.ema_train:.4f} -> {cand_t:.4f})")
            continue

        w, b = w_new, b_new
        gate.commit(cand_t, cand_h)
        if log["ema_held_start"] is None:
            log["ema_held_start"] = float(gate.ema_held)
        log["ema_held_end"] = float(gate.ema_held)
        log["accepted"] += 1
        if log["gap_first"] is None:
            log["gap_first"] = gap_after
            log["heldout_first"] = loss_h_after
        log["gap_last"] = gap_after
        log["heldout_last"] = loss_h_after
        log["train_last"] = float(train_after)
        if verbose and step % 50 == 0:
            print(f"  step {step:4d} accepted  loss {loss_b:.4f}  heldout {loss_h_after:.4f}  "
                  f"gap {gap_after:+.4f}  ema_train {gate.ema_train:.4f} ema_held {gate.ema_held:.4f}  "
                  f"agreement {agreement:.3f}")

    return (w, b), log


def main():
    """A smoke test on the exact case Yahya used: 2 + 2 = 4 must be COMPUTED, not looked up.
    Train on operand pairs 0..9, then score on pairs never seen (10..19)."""
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    X = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in pairs], dtype=np.float64)
    y = np.asarray([a + b for a, b in pairs], dtype=np.float64)
    unseen = [(a, b) for a in range(10, 20) for b in range(10, 20)]
    Xu = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in unseen], dtype=np.float64)
    yu = np.asarray([a + b for a, b in unseen], dtype=np.float64)

    print("=" * 96)
    print("UNIFIED TRAINING - one loop, three target sources, the memorisation rule inside it")
    print("=" * 96)
    print("gate in use: GapGate(variant='ema_band', beta=0.5, tol=0.0)  "
          "[calibrated in lens/scripts/gap_gate_calibration.py]")
    print("prediction P6: the pairs corpus is learnable, so the calibrated gate refuses only a minority of the")
    print("               300 steps, and the OLD undertraining (low exact even on trained pairs) persists")
    print("               because 300 steps at lr=0.05 is too few - a harness problem, not a gate problem.")
    print("               MEASURED OUTCOME: the undertraining part was right, the cause was wrong - it was the")
    print("               agreement throttle, not the step count. The smoke test now uses 6000 steps, because")
    print("               with the gate on about two thirds of the steps are refused, so 300 spent 200 of them")
    print("               on refusals; stage 2 below is the controlled 3-seed version.")
    print("prediction P7: at steps=3000 the same loop reaches >90% exact on trained pairs and on unseen 10..19")
    print("               (a+b is exactly representable by the linear readout), so the harness can judge a gate.")
    print("verifier arm, changed 2026-10-21 18:3x (PLAN verifier2): 12000 steps and the NEW DEFAULT rule")
    print("               verifier_mode='probe3', probe_step=1/16, probe_normalise=True, with the binary")
    print("               right/wrong verifier and NO rank oracle (the fallback path). It used to read 9.0%")
    print("               trained / 0.0% unseen at 6000 steps with the old 'hard_quarter' default.")
    print("               Predictions P55a-P55k were written first: lens/runs/UNIFIED-VERIFIER-PREDICTIONS-ADDENDUM4.txt")
    print()
    for mode in ("data", "teacher", "verifier"):
        if mode == "teacher":
            # a teacher that knows the rule: it returns a+b for whatever it is asked
            teacher = lambda Xb: (Xb[:, 0] * 20.0 + Xb[:, 1] * 20.0)
            (w, b), log = train_unified(X, y, mode="teacher", teacher=teacher, steps=6000, seed=0, verbose=False)
        elif mode == "verifier":
            # the verifier says right or wrong: it checks the rounded answer exactly.
            # 2026-10-21 18:3x (PLAN verifier2): the verifier arm now runs at 12000 steps and uses the NEW
            # DEFAULT (verifier_mode='probe3', probe_step=1/16, probe_normalise=True). NO rank oracle is
            # passed, so this exercises the fallback path a user with only a right/wrong verifier would hit:
            # the binary verifier is asked about the three candidates. Before the change this arm read
            # 9.0% trained / 0.0% unseen with 1015 cold-start steps (lens/runs/UNIFIED-RUN-pre-probe3.txt);
            # the 3000-step budget was the reason the corrected rule reached the true slope on one seed of
            # three, so the item's own requirement was to raise it to 12000.
            verifier = lambda pred, yb: np.abs(np.round(pred) - yb) < 0.5
            (w, b), log = train_unified(X, y, mode="verifier", verifier=verifier, steps=12000, seed=0, verbose=False)
        else:
            (w, b), log = train_unified(X, y, mode="data", steps=6000, seed=0, verbose=False)

        exact_seen = float((np.round(X @ w + b) == y).mean())
        exact_unseen = float((np.round(Xu @ w + b) == yu).mean())
        print(f"\nmode '{mode}':")
        print(f"  accepted steps {log['accepted']}  refused for gap {log['refused_gap']}  "
              f"refused for disagreement {log['refused_agreement']}")
        if log["accepted"] == 0:
            print("  NO STEPS ACCEPTED (see the refusal counts above)")
        else:
            print(f"  held-out loss {log['heldout_first']:.5f} -> {log['heldout_last']:.5f}   "
                  f"smoothed held-out (EMA) {log['ema_held_start']:.5f} -> {log['ema_held_end']:.5f}")
        if log.get("partial_credit_steps"):
            print(f"  cold-start partial-credit steps: {log['partial_credit_steps']}")
        print(f"  exact on the pairs it trained on:      {exact_seen*100:.1f}%")
        print(f"  exact on pairs it NEVER saw (10..19):  {exact_unseen*100:.1f}%   "
              f"<- a lookup table scores 0% here")

    # ---- STAGE 2: is the harness strong enough to judge a gate at all? (UNIFIED-TRAINING.md: "Fix the
    # harness before drawing conclusions"). Same loop, more steps, three settings.
    # MEASURED OUTCOME of prediction P7 (which said the model would reach >90% here): P7 was WRONG at the
    # shipped settings - 44.3% exact on trained pairs - and that was the pointer to the real fault:
    # the step was multiplied by max(agreement, 0.05), throttling every honest step ~3x. With the throttle
    # removed the same loop reaches 100.0% on unseen pairs. See lens/scripts/unified_harness_probe.py.
    print("\n" + "-" * 96)
    print("STAGE 2 - the harness check: 3000 steps, 3 seeds, the setting that was shipped vs the fix")
    print("-" * 96)
    for label, gv, amode in (
        ("shipped: throttle on,  gate on ", "ema_band", "floor"),
        ("FIX    : throttle off, gate on ", "ema_band", "one"),
        ("control: throttle off, gate off", "off", "one"),
    ):
        seen_all, unseen_all, refused_all = [], [], []
        for seed in (0, 1, 2):
            g = GapGate(beta=0.5, tol=0.0, variant=gv)
            (w, b), log = train_unified(X, y, mode="data", steps=3000, lr=0.05, seed=seed,
                                        gate=g, agreement_mode=amode, verbose=False)
            seen_all.append(float((np.round(X @ w + b) == y).mean()))
            unseen_all.append(float((np.round(Xu @ w + b) == yu).mean()))
            refused_all.append(log["refused_gap"])
        print(f"  {label} exact on trained pairs {np.mean(seen_all)*100:5.1f}%   "
              f"exact on UNSEEN pairs {np.mean(unseen_all)*100:5.1f}%   "
              f"refused by the gap gate {np.mean(refused_all):6.1f}/3000")
    print("\nOne method. The only difference between the three modes is where the target came from.")


if __name__ == "__main__":
    main()
