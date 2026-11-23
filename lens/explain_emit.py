"""CAN THE MODEL STATE ITS OWN REASON, AS PART OF ITS ANSWER?

WHY THIS EXISTS (2026-11-22): the owner asked for a model that knows what is important and what is not, built inside the
network. Two measurements now bound the question:

  1. Distraction does NOT hurt it: 100.0% on "find the one digit and copy it" at 8, 16 and 32 decoy positions. So the model
     BEHAVES as if it knows what matters - junk does not fool it.
  2. The probe that tried to REPORT which position mattered failed: 77 out of 100 against a bar of 90. So it acts correctly
     but cannot say what it acted on.

That gap is the whole problem, and the fix follows from this project's own discovery: the machine could not add until it was
allowed to WRITE the working out, one symbol at a time. So do not probe it with a read-out head - ASK IT TO WRITE THE ANSWER
DOWN. The model's output is two tokens:

      [which position held the digit] [the digit itself]

If it can emit the position, the explanation is the model's own answer, visible, and checkable by a person - not a probe
bolted on afterwards. If it cannot, then readability for selection tasks needs a different mechanism, and that is worth
knowing too.

  input :  L positions, exactly ONE holds a real digit, the rest are decoys
  output:  the position of that digit, then the digit

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 it states the reason: position accuracy >= 90% on held-out samples.
  P2 it still answers:   digit accuracy >= 95% on held-out samples.
  P3 both right together (the answer is usable as an explanation): >= 85%.
FALSIFIER: position accuracy < 80% -> the model cannot state which input it used even when asked to write it down, and
  "explain by emitting" does not work for selection the way it worked for carrying.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glass_lm import GlassLM, Momentum

N_DIG = 10                     # digit token ids 0..9
N_NOISE = 6                    # decoy token ids
NOISE0 = N_DIG
POS0 = N_DIG + N_NOISE         # position-report tokens: POS0 + p
PAD = POS0 + 64                # plenty of room for wide inputs
# COLLISION FIXED (2026-11-22 16:22): the cue and reason tokens were placed at PAD-3..PAD, so REASON_SMALL shared an id with the
# padding token. A slot that was never written would then be indistinguishable from a written "smaller" reason, and the model could
# score the reason for free. They now sit ABOVE the padding id, with V extended to cover them.
CUE_BIG, CUE_SMALL = PAD + 1, PAD + 2
CUE_FIRST, CUE_LAST = PAD + 3, PAD + 4
REASON_BIG, REASON_SMALL = PAD + 5, PAD + 6
REASON_FIRST, REASON_LAST = PAD + 7, PAD + 8
V = PAD + 9

L_IN = 12                      # input positions
TASK = "copy1"
N_RULES = 2
ANS_LEN = 2                    # how many tokens the answer is; max2write writes the comparison out first, so it is 6
T = L_IN + 1                   # the model reads L_IN tokens and predicts 2 answer tokens


def make_seq(rng, l_in):
    """max2write: report both digits in position order, then the position and value of the larger one."""
    """copy1: one digit, report where it is.  max2: two digits, report where the LARGER one is and what it is."""
    toks = rng.integers(NOISE0, NOISE0 + N_NOISE, size=l_in)
    if TASK == "trigword":
        # THE REASON IS A WORD, NOT A SPECIAL TOKEN (2026-11-22 19:38). In cuechoose the cue is a dedicated token id, which is a hint
        # in itself. Here the trigger word is drawn from the SAME id pool as the decoys, so among the non-digit tokens one of them means
        # something and the rest are noise that looks identical. The model must find it and NAME it as its reason.
        p1, p2 = sorted(int(x) for x in rng.choice(l_in, size=2, replace=False))
        d1, d2 = int(rng.integers(0, N_DIG)), int(rng.integers(0, N_DIG))
        while d2 == d1:
            d2 = int(rng.integers(0, N_DIG))
        toks[p1], toks[p2] = d1, d2
        trig = int(rng.integers(NOISE0, NOISE0 + 4))        # one of four ordinary words means something
        slot = int(rng.integers(0, l_in))
        while slot in (p1, p2):
            slot = int(rng.integers(0, l_in))
        toks[slot] = trig
        # decoys elsewhere must NOT be one of the four trigger words, or the trigger would be ambiguous
        for i in range(l_in):
            if i in (p1, p2, slot):
                continue
            v = int(toks[i])
            if NOISE0 <= v < NOISE0 + 4:
                toks[i] = NOISE0 + 4 + (v - NOISE0) % max(1, N_NOISE - 4)
        rule = trig - NOISE0
        if rule == 0:
            pm, dm, reason = (p1, d1, REASON_BIG) if d1 > d2 else (p2, d2, REASON_BIG)
        elif rule == 1:
            pm, dm, reason = (p1, d1, REASON_SMALL) if d1 < d2 else (p2, d2, REASON_SMALL)
        elif rule == 2:
            pm, dm, reason = p1, d1, REASON_FIRST
        else:
            pm, dm, reason = p2, d2, REASON_LAST
        return toks.astype(np.int64), {"write": [(pm, dm)], "p": pm, "d": dm, "reason": reason}
    if TASK.startswith("cuechoose"):
        # ONE cue token says which rule to follow; the answer is the larger or the smaller of two digits, by the cue.
        p1, p2 = sorted(int(x) for x in rng.choice(l_in, size=2, replace=False))
        d1, d2 = int(rng.integers(0, N_DIG)), int(rng.integers(0, N_DIG))
        while d2 == d1:
            d2 = int(rng.integers(0, N_DIG))
        toks[p1], toks[p2] = d1, d2
        rule = int(rng.integers(0, N_RULES))
        cue = (CUE_BIG, CUE_SMALL, CUE_FIRST, CUE_LAST)[rule]
        slot = int(rng.integers(0, l_in))
        while slot in (p1, p2):
            slot = int(rng.integers(0, l_in))
        toks[slot] = cue
        if rule == 0:
            pm, dm, reason = (p1, d1, REASON_BIG) if d1 > d2 else (p2, d2, REASON_BIG)
        elif rule == 1:
            pm, dm, reason = (p1, d1, REASON_SMALL) if d1 < d2 else (p2, d2, REASON_SMALL)
        elif rule == 2:
            pm, dm, reason = p1, d1, REASON_FIRST
        else:
            pm, dm, reason = p2, d2, REASON_LAST
        return toks.astype(np.int64), {"write": [(pm, dm)], "p": pm, "d": dm, "reason": reason}
    if TASK == "max3write":
        p1, p2, p3 = sorted(int(x) for x in rng.choice(l_in, size=3, replace=False))
        vals = []
        while len(set(vals)) < 3:
            vals = [int(v) for v in rng.integers(0, N_DIG, size=3)]
        d1, d2, d3 = vals
        toks[p1], toks[p2], toks[p3] = d1, d2, d3
        trip = [(p1, d1), (p2, d2), (p3, d3)]
        pm, dm = max(trip, key=lambda t: t[1])
        return toks.astype(np.int64), {"write": trip + [(pm, dm)], "p": pm, "d": dm}
    if TASK == "max2write":
        p1, p2 = sorted(int(x) for x in rng.choice(l_in, size=2, replace=False))
        d1, d2 = int(rng.integers(0, N_DIG)), int(rng.integers(0, N_DIG))
        while d2 == d1:
            d2 = int(rng.integers(0, N_DIG))
        toks[p1], toks[p2] = d1, d2
        pm, dm = (p1, d1) if d1 > d2 else (p2, d2)
        return toks.astype(np.int64), {"write": [(p1, d1), (p2, d2), (pm, dm)], "p": pm, "d": dm}
    if TASK in ("max2", "max2write"):
        p1, p2 = rng.choice(l_in, size=2, replace=False)
        d1, d2 = int(rng.integers(0, N_DIG)), int(rng.integers(0, N_DIG))
        while d2 == d1:                       # equal digits make "the larger" ambiguous, so exclude them
            d2 = int(rng.integers(0, N_DIG))
        toks[int(p1)], toks[int(p2)] = d1, d2
        if d1 > d2:
            return toks.astype(np.int64), int(p1), d1
        return toks.astype(np.int64), int(p2), d2
    p = int(rng.integers(0, l_in))
    d = int(rng.integers(0, N_DIG))
    toks[p] = d
    return toks.astype(np.int64), {"write": [(p, d)], "p": p, "d": d}


def to_batch(seqs, l_in, T_):
    """tokens/targets; every written token is supervised, so a written-out comparison is graded too."""
    n = len(seqs)
    an = ANS_LEN
    full = np.full((n, l_in + an + 1), PAD, dtype=np.int64)
    for i, (toks, ans) in enumerate(seqs):
        full[i, :l_in] = toks
        if TASK.startswith("cuechoose") or TASK == "trigword":
            full[i, l_in] = ans["reason"]          # the model writes WHICH RULE it used, first
            full[i, l_in + 1] = POS0 + ans["p"]
            full[i, l_in + 2] = ans["d"]
            continue
        for t, (p, d) in enumerate(ans["write"]):
            full[i, l_in + 2 * t] = POS0 + p
            full[i, l_in + 2 * t + 1] = d
    x = full[:, :T_]
    y = full[:, 1:T_ + 1].copy()
    mask = np.zeros_like(y, dtype=bool)
    for t in range(an):
        mask[:, l_in - 1 + t] = True
    y[~mask] = -1
    return x, y


def ans_has_reason(ans, got):
    return got == ans.get("reason")


def expected_reason(toks, l_in):
    """Which rule the input asked for - readable straight off the cue token."""
    for t in toks[:l_in]:
        if int(t) == CUE_BIG:
            return REASON_BIG
        if int(t) == CUE_SMALL:
            return REASON_SMALL
    return None


def expected_answer(toks, l_in):
    """The right answer, derived from the TOKENS THEMSELVES, for whichever task is running.

    WHY THIS EXISTS (2026-11-22 14:12): the first version of the perturbation test ASSUMED the answer after changing the input -
    "swap the digit and it should report the new value at the same position". That assumption is true for copy1 (one digit) and
    FALSE for max2: swapping the larger digit can make the OTHER digit the larger one, so the correct position changes too. The
    test therefore reported 72.5%/66.5% and blamed the model for my arithmetic. Expected answers are now computed from the
    perturbed tokens by the task's own definition, so the test cannot be wrong about what the right answer is."""
    digs = [(i, int(t)) for i, t in enumerate(toks[:l_in]) if 0 <= int(t) < N_DIG]
    if not digs:
        return None, None
    if TASK == "max2":
        i, v = max(digs, key=lambda t: t[1])
        return i, v
    i, v = digs[0]
    return i, v


def decode(lm, toks, l_in):
    """Free-running: read the input, then emit the whole answer, feeding each token back."""
    ctx = list(toks[:l_in])
    out = []
    for _ in range(ANS_LEN):
        logits, _ = lm.forward(np.array(ctx, dtype=np.int64)[None, :])
        nxt = int(np.argmax(logits[0, len(ctx) - 1, :]))
        out.append(nxt)
        ctx.append(nxt)
    return out


def main():
    # global MUST be the first statement: an earlier reference (even inside a default=L_IN) makes this a SyntaxError, and I hit
    # this exact trap once before today. The compile gate caught it immediately this time, before any launch.
    global L_IN, T, TASK, ANS_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pos", type=int, default=L_IN)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--task", default="copy1", choices=["copy1", "max2", "max2write", "max3write", "cuechoose", "cuechoose4", "trigword"],
                    help="copy1 = one digit, report where it is. max2 = TWO digits, report the position of the LARGER and its value "
                         "(needs selection AND comparison, so it is the honest next rung after one position)")
    a = ap.parse_args()

    L_IN = a.n_pos
    TASK = a.task
    # max2write: the model WRITES BOTH DIGITS AND THEIR POSITIONS before answering, the way carrying is written out. This is the
    # representation change the capacity test pointed at: same parameters, same steps, same bar, only how the choice is recorded.
    # cuechoose: the model must WRITE WHICH RULE APPLIED, then the position and value of the digit that rule picks.
    N_RULES = 4 if TASK in ("cuechoose4", "trigword") else 2
    ANS_LEN = 6 if TASK == "max2write" else (8 if TASK == "max3write" else (3 if (TASK.startswith("cuechoose") or TASK == "trigword") else 2))
    T = L_IN + ANS_LEN - 1
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    lm = GlassLM(V, d=a.d, layers=a.layers, n_heads=a.heads, d_mlp=2 * a.d, T=T, seed=a.seed, emb_rank=4)
    print("explain-by-emitting: task %s, %d positions, %d decoy kinds" % (TASK, L_IN, N_NOISE))
    if TASK == "max2":
        print("two real digits; the model must write [position of the LARGER][the larger value]")
    print("the model must WRITE the position AND the digit, two tokens, no probe head")
    print("model: %s parameters, context %d" % (format(lm.n_params(), ","), T))

    ev = [make_seq(np.random.default_rng(10_000 + i), L_IN) for i in range(a.n_eval)]
    opt = Momentum(lm.p, lr=a.lr, momentum=0.9)
    rr = np.random.default_rng(a.seed + 1)
    for step in range(1, a.steps + 1):
        seqs = [make_seq(rr, L_IN) for _ in range(a.batch)]
        x, y = to_batch(seqs, L_IN, T)
        loss, acc, grads = lm.loss_and_grads(x, y, unk=-1)
        opt.step(grads)
        if step % 500 == 0 or step == 1:
            print("step %5d/%d  loss %.4f  token acc %.1f%%  %.0fs"
                  % (step, a.steps, loss, 100 * acc, time.time() - t0), flush=True)

    # ---------- verdict ----------
    pos_ok = dig_ok = both = reason_ok = 0
    shown = 0
    for toks, ans in ev:
        p, d = ans["p"], ans["d"]
        out = decode(lm, toks, L_IN)
        if TASK.startswith("cuechoose") or TASK == "trigword":
            gp, gd = out[1] - POS0, out[2]
            if ans_has_reason(ans, out[0]) is False:
                reason_ok_local = 0
            else:
                reason_ok_local = 1
            reason_ok += reason_ok_local
        else:
            # the ANSWER is the last pair written; earlier pairs are the working out
            gp, gd = out[-2] - POS0, out[-1]
        ok_p, ok_d = (gp == p), (gd == d)
        pos_ok += ok_p
        dig_ok += ok_d
        both += ok_p and ok_d
        if shown < 4:
            shown += 1
            if TASK.startswith("cuechoose") or TASK == "trigword":
                want_reason = expected_reason(toks, L_IN)
                print("  cue=%s | chose digit %d at position %d | model wrote reason %s position %s digit %s"
                      % ("BIG" if want_reason == REASON_BIG else "SMALL", d, p,
                         "BIG" if out[0] == REASON_BIG else ("SMALL" if out[0] == REASON_SMALL else "OTHER"),
                         gp if 0 <= gp < L_IN else "OUT-OF-RANGE", gd))
            else:
                print("  input %s | digit %d at position %d | model said position %s digit %s"
                      % (list(toks[:L_IN]), d, p, gp if 0 <= gp < L_IN else "OUT-OF-RANGE", gd))
    n = len(ev)
    pa, da, ba = 100 * pos_ok / n, 100 * dig_ok / n, 100 * both / n
    if TASK.startswith("cuechoose") or TASK == "trigword":
        ra = 100 * reason_ok / n
        # REASON HONESTY (2026-11-22 18:47): stating the cue is not the same as reporting the act. A model could parrot the cue while
        # picking a digit the cue did not ask for, and the reason would still be "correct". This counts how often the stated reason
        # MATCHES THE DIGIT ACTUALLY CHOSEN, which is the stronger claim and the one a reader relies on.
        agree = 0
        for toks, ans in ev:
            out = decode(lm, toks, L_IN)
            digs = [(i, int(t)) for i, t in enumerate(toks[:L_IN]) if 0 <= int(t) < N_DIG]
            if len(digs) < 2:
                continue
            mx = max(digs, key=lambda t: t[1])[1]
            mn = min(digs, key=lambda t: t[1])[1]
            said = out[0]
            picked = out[2]
            if said == REASON_BIG and picked == mx:
                agree += 1
            elif said == REASON_SMALL and picked == mn:
                agree += 1
            elif N_RULES == 4 and said == REASON_FIRST and out[1] - POS0 == digs[0][0]:
                agree += 1
            elif N_RULES == 4 and said == REASON_LAST and out[1] - POS0 == digs[-1][0]:
                agree += 1
        print("CONSISTENCY: stated reason matches the digit actually written in %.1f%% of examples (n=%d)"
              % (100 * agree / n, n))
        print("P3 reason is HONEST about the act (consistency >= 95%%): %s  (%.1f%%)"
              % ("MET" if 100 * agree / n >= 95 else "NOT MET", 100 * agree / n))
        print("RESULT reason stated correctly = %.1f%% | position = %.1f%% | value = %.1f%% | all three = %.1f%% (n=%d)"
              % (ra, pa, da, ba, n))
        print("P1 states WHICH RULE it used (reason >= 95%%): %s  (%.1f%%)" % ("MET" if ra >= 95 else "NOT MET", ra))
        print("OUTCOME: %s" % ("MODEL_WRITES_ITS_OWN_REASON" if ra >= 95 and ba >= 90 else "CANNOT_STATE_ITS_RULE"))
        print("EXPLAIN-EMIT COMPLETE")
        return

    # ---------- SECOND MEASUREMENT: does it actually follow the input, or did it find a shortcut? ----------
    # A clean 100.0% on the first try is exactly the shape that hid the ternary defect this morning, so before any of this
    # is believed the model has to survive changing the input in a way whose correct answer is known:
    #   MOVED : the same digit put at a DIFFERENT position -> the emitted position must move with it
    #   SWAPPED: the same position holding a DIFFERENT digit -> the emitted digit must change, the position must not
    # If it cannot do both, its 100.0% was reading something other than where the digit is.
    rng2 = np.random.default_rng(777)
    moved_ok = swap_ok = swap_pos_ok = 0
    skipped = 0
    for toks, ans_ in ev:
        p, d = ans_["p"], ans_["d"]
        q = int((p + 1 + int(rng2.integers(0, L_IN - 1))) % L_IN)      # a different position, guaranteed
        mv = toks.copy()
        mv[p] = int(rng2.integers(NOISE0, NOISE0 + N_NOISE))           # the old spot becomes a decoy
        mv[q] = d
        ep, ev_ = expected_answer(mv, L_IN)                            # computed, not assumed
        out = decode(lm, mv, L_IN)
        moved_ok += (out[0] - POS0 == ep) and (out[1] == ev_)
        sw = toks.copy()
        nd = (d + 1 + int(rng2.integers(0, N_DIG - 1))) % N_DIG        # a different digit value
        sw[p] = nd
        ep2, ev2 = expected_answer(sw, L_IN)
        if ep2 is None:
            skipped += 1
            continue
        out2 = decode(lm, sw, L_IN)
        swap_ok += (out2[1] == ev2)
        swap_pos_ok += (out2[0] - POS0 == ep2)
    n_sw = max(n - skipped, 1)
    print()
    print("SECOND MEASUREMENT (the input is changed, the right answer is known):")
    print("  digit MOVED to another position -> model follows it: %.1f%%" % (100 * moved_ok / n))
    print("  digit value SWAPPED -> model reports the correct value:   %.1f%%" % (100 * swap_ok / n_sw))
    print("  digit value SWAPPED -> model reports the correct position: %.1f%%" % (100 * swap_pos_ok / n_sw))
    tracks = (moved_ok / n >= 0.95) and (swap_ok / n_sw >= 0.95) and (swap_pos_ok / n_sw >= 0.95)
    print("  VERDICT: %s" % ("it really follows the input" if tracks else
                             "IT DOES NOT FOLLOW THE INPUT - the headline 100% cannot be believed"))
    print()
    print("RESULT position stated correctly = %.1f%% | digit correct = %.1f%% | both = %.1f%% (n=%d)"
          % (pa, da, ba, n))
    print("P1 states the reason (position >= 90%%):    %s  (%.1f%%)" % ("MET" if pa >= 90 else "NOT MET", pa))
    print("P2 still answers (digit >= 95%%):           %s  (%.1f%%)" % ("MET" if da >= 95 else "NOT MET", da))
    print("P3 usable explanation (both >= 85%%):       %s  (%.1f%%)" % ("MET" if ba >= 85 else "NOT MET", ba))
    if not tracks:
        print("OUTCOME: EXPLANATION_NOT_TRACKING")
        print("EXPLAIN-EMIT COMPLETE")
        return
    if pa < 80:
        print("FALSIFIER (position < 80%):                 FIRED - explain-by-emitting does not work for selection")
        print("OUTCOME: CANNOT_STATE_REASON")
    elif ba >= 85:
        print("FALSIFIER (position < 80%):                 not fired")
        print("OUTCOME: EXPLAINS_BY_EMITTING")
    else:
        print("FALSIFIER (position < 80%):                 not fired")
        print("OUTCOME: PARTIAL_REASON")
    print("EXPLAIN-EMIT COMPLETE")


if __name__ == "__main__":
    main()
