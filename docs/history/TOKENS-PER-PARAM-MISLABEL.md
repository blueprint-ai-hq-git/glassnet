# The "tokens per parameter" line was windows per parameter - every figure low by 32x

2026-11-07 11:00 IST. `lens/scripts/train_chat.py:830` printed

    tpp = len(Xtr) / max(net.n_params(), 1)
    log(f"  tokens per parameter: {tpp:.2f}  (...)")

but `len(Xtr)` is the number of **windows**, not tokens: `Xtr` comes from
`chat_data.make_windows(ids, bounds, split, a.window, ...)` and is an array of shape `[n_windows, window]`
(`train_chat.py:546`, `:554`). So every "tokens per parameter" figure in every log of this project is low by
exactly the window length, 32 - and the accompanying verdict ("well trained" > 10, "adequate" > 4, else
"UNDER-TRAINED") was computed on the same wrong quantity.

## What the real numbers are

    arm                          printed   true tokens/param   verdict the trainer now prints
    d256 @ 20k windows             0.01         0.17           UNDER-TRAINED
    d256 @ 200k windows            0.05         1.68           UNDER-TRAINED
    d256 @ 500k windows            0.13         4.20           adequate      <-- the current arm
    d512 @ 4M windows (TIED-D512)  0.26         8.22           adequate      <-- printed UNDER-TRAINED
    d512 @ 200k windows (proposed) 0.01         0.41           UNDER-TRAINED

Params: 3,811,534 for the d=256 family (`DATASCALE-HONEST.txt` epoch lines), **15,567,566** for d=512, K=36
(read from `TIED-D512.txt`, `model: 15,567,566 params (d=512, K=36)`).

## Why it matters, and why it is not cosmetic

* **A printed verdict was wrong.** The `TIED-D512` rung printed `tokens per parameter: 0.26 (UNDER-TRAINED)`;
  its true figure is **8.22 - "adequate"**. Any conclusion of the form "d=512 was under-trained at 4M windows"
  rests on a number that is wrong by 32x.
* **It explains today's arm.** 500k windows at d=256 is **4.20 tokens/param**, the first rung in this family
  that crosses the project's own "adequate" line - which is consistent with its first epoch already reaching
  VAL 20.03% (`DATASCALE-500K.txt` ep 1), where the 200k arm (1.68) took three epochs to reach 17.43%.
* **It sharpens the width decision.** The proposed d=512 run at the 200k rung would be **0.41 tokens/param**,
  an order of magnitude below "adequate". Declining that arm is now a measured judgement, not a guess.

## This is not a new discovery - it is an old one that was never fixed

`AUTOPILOT-LOG.md:4552` already records it, as a *prediction* error: "[BUG-PREDICTION-CLAIMS-A-PRINTED-NUMBER-
THAT-IS-NOT-PRINTED] P110e claims the trainer prints 6.6 tokens/parameter when it prints windows/params =
0.36". The prediction was corrected; the code was not, and every run since that entry carries the wrong label.

## The fix

The label now computes what it claims, and the old ratio is printed on the same line so historical arms remain
comparable without anyone having to remember this note:

    tpp = len(Xtr) * int(getattr(a, "window", 1)) / max(net.n_params(), 1)
    wpp = len(Xtr) / max(net.n_params(), 1)
    log(f"  tokens per parameter: {tpp:.2f}  (...)"
        f"   [windows/params {wpp:.3f}; windows {len(Xtr):,} x {window} tokens]")

Verified: `py_compile` clean; the five cases above recomputed independently; no script or grader parses the old
string (only prose in `LOG.md`, `ARCHITECTURE-FIX.md`, `AUTOPILOT-LOG.md`), so nothing downstream breaks.
The 500k arm in flight is unaffected - Python loaded the file at launch, and its log keeps the old line.

**Old logs are not rewritten.** They are the record; this note is the correction attached to them.

## ADDENDUM 2026-11-07 15:3x: the figure counts token INSTANCES, and with stride 16 half of each window duplicates its neighbour

The fix above made the line count tokens instead of windows. There is a second, smaller unit question under it.
`chat_data.py:166` sets `stride = window // 2`, so consecutive windows share **half their tokens**. The printed
figure is therefore window-instances x window / params, not distinct corpus tokens seen.

For the 37.37% headline (16,900,000 windows x 32 = **540,800,000 token-instances**, corpus 294,225,090 tokens,
15,567,566 params):

    instances / corpus       = 1.838   -> each token is covered by ~1.84 windows
    printed figure, 1 epoch  = 34.74   tokens/param  (instance count)
    distinct tokens/param    = 18.90   (= corpus / params, i.e. one full pass of the corpus)
    six epochs               = 208.4 instances/param, 113.4 distinct tokens/param

So the headline is **well trained by either count** (18.9 and 113.4 both exceed the tool's 10 threshold), which is
why the earlier correction's conclusion stands. But the thresholds are being applied to numbers roughly **twice**
the distinct-token coverage: the d=256/500k arm prints `4.20 (adequate)` while its distinct coverage is about
**2.1 tokens/param**, and the 20k arm prints `0.17` for about `0.084`. Anyone reasoning about over- or
under-training from that line should halve it first.

What was NOT changed: the thresholds themselves. Moving them would silently reclassify every historical run, and
they are his call, not mine.
