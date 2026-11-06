"""train_chat.py — train a model that TALKS, then make it talk so it can be judged.

THE POINT
---------
Perplexity does not tell you whether a model can hold a conversation. The only honest test is to
give it a prompt and read what it says. So this trains, then GENERATES, and prints the replies. If
the output is gibberish, that is what gets reported.

SCALE, AND WHY THIS SIZE
------------------------
The corpus holds 13.3M tokens. Chinchilla-optimal is ~20 tokens per parameter, so a 1M-parameter
model is well trained here (13 tokens/param) while a 10M model would be UNDER-trained (1.3
tokens/param). The data is the binding constraint, not the GPU, so the size follows the data.

ARCHITECTURE
------------
The readable one: embedding -> named concept layer -> answer layer. The concepts are conversational
(see chat_concepts.py) and the fidelity audit applies to them exactly as it does elsewhere.
"""
import argparse
import json
import os
import pickle
import re
import sys
import signal

def _open_log(path):
    """Open an append log with retries. THIS DRIVE IS NTFS (ntfs3) and has taken two power cuts; ntfs3 has
    returned OSError 22 (EINVAL) on append and killed a whole training arm (ours_lr100_s1, exit=1, seed lost).
    Retry briefly, then re-raise, so a transient filesystem error costs a second, not a run."""
    last = None
    for attempt in range(5):
        try:
            return open(path, "a")
        except OSError as e:
            last = e
            time.sleep(0.4 * (attempt + 1))
    raise last

import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import chat_concepts                                       # noqa: E402
import chat_data                                           # noqa: E402
import lens                                                # noqa: E402
import lens.tensor as T                                    # noqa: E402
from lens import backend as B                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402
from lens.model import Dense                               # noqa: E402
from lens.opt import RelativeGrad                          # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "chat")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


class ChatNet:
    """Thin wrapper over the library's LensNet: the architecture IS the library's architecture.

    The first version reimplemented the model here and then failed three times in a row on
    attributes that lens.save expects (seed, lam_act, lam_w). Reimplementing what the library
    already provides was the mistake -- LensNet has the concept layer, the concept loss, the
    sparsity term, and the save/load contract. This delegates to it.
    """

    def __init__(self, vocab, concepts, d=128, window=8, seed=0, lam_concept=0.3,
                 lam_act=0.0, device="gpu", attention=False, mtp=1, direct=False, direct_rank=16, direct_mode="lowrank"):
        T.set_device(device)
        self._net = lens.Net(vocab=vocab, concepts=concepts, d=d, window=window, mtp=mtp,
                             seed=seed, lam_concept=lam_concept, lam_act=lam_act,
                             attention=attention, device=device, direct=direct,
                             direct_rank=direct_rank, direct_mode=direct_mode)
        self.concepts = self._net.concepts
        self.K, self.V, self.D, self.W = self._net.K, self._net.V, self._net.D, self._net.W
        self.seed = int(seed)
        self.lam_concept, self.lam_act, self.lam_w = lam_concept, lam_act, getattr(self._net,
                                                                                 "lam_w", 1.0)
        self.emb, self.concept, self.answer = self._net.emb, self._net.concept, self._net.answer

    def forward(self, X):
        out = self._net.forward(X)
        return out["concepts"], out["logits"]

    def loss(self, X, Y, weights=None, y2=None, concept_ids=None):
        # `concept_ids` (2026-10-25 self-wake): passed straight through to the library's loss. The
        # self-context (exposure-bias) step gives the CE term the rolled-in window but keeps the CONCEPT
        # target on the original one. Default None = unchanged for every caller.
        return self._net.loss(X, Y, weights=weights, y2=y2, concept_ids=concept_ids)

    def forward_arrays(self, X):
        return self._net.forward_arrays(X)

    def parameters(self):
        return self._net.parameters()

    def n_params(self):
        return int(sum(p.data.size for p in self.parameters()))


def concept_targets(X, concepts, K):
    """1 where a concept's tokens appear in the window, else 0. Targets come from TRAIN windows."""
    t = np.zeros((len(X), K), dtype=np.float32)
    for k, c in enumerate(concepts):
        t[:, k] = np.isin(X, c.ids).any(axis=1).astype(np.float32)
    return t


def concept_bce(h, tgt, w, eps=1e-7):
    """Balanced BCE on the concept layer, built as a registered OP.

    The project's engine has no closures by design (the L1 gate exists to prevent the reference
    cycles the old engine leaked), so the node is constructed with an `_op` name and the backward
    is looked up in the shared op table. `w` is the per-concept positive weight derived from the
    declared support, so a rare concept is not drowned out by a common one.
    """
    hd = h.data
    bce = -(w * tgt * B.xp.log(hd + eps) + (1.0 - tgt) * B.xp.log(1.0 - hd + eps))
    val = float(B.to_cpu(bce.mean()))
    return T.Tensor(B.to_dev(np.asarray(val, dtype=np.float32)),
                    requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                    _saved={"hd": hd, "tgt": tgt, "w": w, "n": bce.size})


def target_ce(net, X, Y, W=None, bs=256):
    """Mean per-target cross-entropy of the model on these windows, NO GRADIENTS, no graph.

    This is the same quantity the epoch line's VAL and MASKED columns are built from (per-target
    cross-entropy on the target token, restricted to the targets that carry weight when a mask is given), so
    the gate's number and the reported number are the same kind of number. It is the trainer's OWN loss
    definition rather than a new one, which matters: the 2026-10-20 teacher-mode bug was a probe scored
    against a target the model was never asked to match.
    """
    tot, n = 0.0, 0
    for s in range(0, len(X), bs):
        idx = np.arange(s, min(s + bs, len(X)))
        lg = np.asarray(net.forward_arrays(X[idx])["logits"], dtype=np.float64)
        mx = lg.max(axis=1, keepdims=True)
        logp = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
        ce = -logp[np.arange(len(idx)), Y[idx]]
        if W is not None:
            w = W[idx] > 0
            if not w.any():
                continue
            tot += float(ce[w].sum())
            n += int(w.sum())
        else:
            tot += float(ce.sum())
            n += len(idx)
    return tot / max(n, 1)


def grad_flat(net):
    """Every parameter's gradient as its own flat float64 array (None where there is no gradient)."""
    out = []
    for p in net.parameters():
        g = p.grad
        out.append(None if g is None else np.asarray(B.to_cpu(g), dtype=np.float64).reshape(-1))
    return out


def agreement_coherence(net, X, Y, W=None, y2=None, groups=8):
    """A = ||sum_g grad_g|| / sum_g ||grad_g|| over `groups` equal sub-batches of these windows.

    LEARN-NOT-MEMORISE.md section 2b, as a number. The batch is cut into `groups` sub-batches; each gets its
    own forward+backward through the SAME loss the training uses (weights and the mtp target included), and
    the sub-batch gradients are compared: A = 1 when they all point the same way, A ~ 1/sqrt(groups) when they
    are mutually independent (0.354 at 8), A = 0 when they cancel.

    RETURNS (A, acc, G): the number, the per-parameter SUM of the sub-batch gradients (so a caller can check
    what that sum is worth as a batch gradient - see P60m), and how many sub-batches were actually used.

    COST, STATED: this re-uses p.grad, so the caller MUST snapshot and restore the full-batch gradient around
    it. It is `groups` forward+backward passes over the SAME windows the batch already covers, i.e. one extra
    pass over the data plus Python overhead - not `groups` times the work of a step.
    """
    n = len(X)
    if n == 0:
        return 0.0, None, 0
    G = int(max(1, min(int(groups), n)))
    edges = np.linspace(0, n, G + 1).astype(int)
    acc, den, used = None, 0.0, 0
    for gi in range(G):
        a0, a1 = int(edges[gi]), int(edges[gi + 1])
        if a1 <= a0:
            continue
        used += 1
        for p in net.parameters():
            p.grad = None
        total, _parts = net.loss(X[a0:a1], Y[a0:a1],
                                 weights=(W[a0:a1] if W is not None else None),
                                 y2=(y2[a0:a1] if y2 is not None else None))
        total.backward()
        gs = grad_flat(net)
        if acc is None:
            acc = [None if g is None else np.zeros(g.shape, dtype=np.float64) for g in gs]
        for i, g in enumerate(gs):
            if g is None:
                continue
            acc[i] += g
            den += float(np.sqrt(float(np.dot(g, g))))
    if acc is None or den <= 0.0:
        return 0.0, acc, used
    num = 0.0
    for g in acc:
        if g is not None:
            num += float(np.dot(g, g))
    return float(np.sqrt(num) / den), acc, used


def _write_mid(path, *, net, opt, gate, agate, ep, step_next, acc, gstep, astep, gstats, astats,
               best_val, best_ep, cfg_hash, train_windows):
    """Write the MID-EPOCH checkpoint: the weights AND every piece of state a continuation needs.

    WHY IT IS ONE FILE (2026-10-23): the weights and the step count must agree. Two files can be torn apart
    by a shutdown between the two writes, and a run that resumes from newer weights with an older step count
    (or the reverse) is no longer the run it claims to be. One pickle written to a temporary name and moved
    into place with os.replace() is atomic, so the file on disk always matches itself.

    WHAT IT CARRIES, and why each part is there: the step index (so the epoch continues where it stopped),
    the epoch's running totals (so the epoch line reports the WHOLE epoch and not just the tail), the step
    rule's state (relgrad's step counter drives the decay schedule), both gates' state (their EMAs decide the
    next verdict), best_val/best_ep (without them a resumed run would overwrite a better earlier *_best.npz
    with a worse later epoch -- the exact mistake best-epoch saving was added to stop), and cfg_hash.
    STATED LIMIT: the gates' history lists are capped at the last 2000 entries. That is exact for the
    shipped detectors (ema_band / vs_probe read only the EMA); it would only bite the lagged trend detectors
    (--gate-lag above 2000), which no run uses. Written here rather than left to be discovered.
    """
    params = {}
    for pt in net._net.parameters():
        params[pt.name] = np.ascontiguousarray(B.to_cpu(pt.data))
    state = {
        "version": 1,
        "written_at": time.time(),
        "epoch": int(ep),
        "step": int(step_next),          # steps COMPLETED in this epoch; the resume starts at this step
        "acc": dict(acc),
        "gstep": int(gstep), "astep": int(astep),
        "gate_stats": dict(gstats), "astats": dict(astats),
        "best_val": best_val, "best_ep": best_ep,
        "cfg_hash": cfg_hash, "train_windows": int(train_windows),
        "params": params,
        "opt": ({"steps": int(opt.steps), "lr": float(opt.lr), "decay": float(opt.decay),
                 "mode": opt.mode, "momentum": float(opt.momentum), "scale": bool(opt.scale),
                 "ema": {k: np.asarray(v) for k, v in opt._ema.items()},
                 "mom": {k: np.asarray(v) for k, v in opt._mom.items()}} if opt is not None else None),
        "gate": ({"ema_train": gate.ema_train, "ema_held": gate.ema_held, "best_held": gate.best_held,
                  "hist_t": [float(v) for v in list(gate.hist_t)[-2000:]],
                  "hist_h": [float(v) for v in list(gate.hist_h)[-2000:]],
                  "checked": int(gate.checked), "refused": int(gate.refused),
                  "gap_first": gate.gap_first, "gap_last": gate.gap_last}
                 if gate is not None else None),
        "agate": ({"ema": agate.ema, "a_first": agate.a_first, "a_last": agate.a_last,
                   "checked": int(agate.checked), "refused": int(agate.refused)}
                  if agate is not None else None),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=4)
    os.replace(tmp, path)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--lam-act", type=float, default=0.0,
                    help="penalty on mean(|concept activation|). 0.0 = off (every run so far). Raising it "
                         "pressures units to stay SILENT unless their own evidence is present, which is "
                         "the fix for the measured over-firing: 14.10 units firing per window and 2334 "
                         "false fires in a random draw of 200 held-out windows.")
    ap.add_argument("--max-gap", type=float, default=0.0,
                    help="STOP TRAINING when (train accuracy - held-out accuracy) exceeds this. The rule "
                         "is that the model must LEARN, never MEMORISE, so a widening gap is treated as a "
                         "failure signal instead of progress. 0 disables the gate.")
    ap.add_argument("--lr-decay", type=float, default=0.0,
                    help="0 = constant lr. Otherwise multiply lr by (1 - decay*progress).")
    ap.add_argument("--opt", choices=["sgd", "relgrad"], default="sgd",
                    help="THE STEP RULE. sgd = the hand-rolled rule this file has always used "
                         "(p.data = p.data - lr_now*p.grad), bit-identical to before this flag existed. "
                         "relgrad = OUR rule (lens/opt.py RelativeGrad): the update of every tensor has "
                         "norm exactly --opt-lr in weight space, whatever its size or gradient scale, so "
                         "one setting is meant to transfer across model sizes instead of being hunted per "
                         "size. Measured on the numeric path (lens/runs/NUMERIC-PATH-OPT.txt): a CONSTANT "
                         "normalised step is non-monotonic in the step size and does not even fit a "
                         "5-parameter regression; with --opt-decay-final 0.02 it answers unseen operand "
                         "pairs exactly at EVERY step size tried, 0.005 through 0.5.")
    ap.add_argument("--opt-lr", type=float, default=None,
                    help="step size for --opt relgrad: the SIZE of each tensor's update in weight space, "
                         "NOT a multiplier on an unbounded gradient. Default: --lr. It is NOT on the same "
                         "scale as --lr for SGD -- run --probe-grads to see the matched value on this net.")
    ap.add_argument("--opt-momentum", type=float, default=0.0,
                    help="relgrad only: EMA on the gradient before normalising (0 = off).")
    ap.add_argument("--opt-decay-final", type=float, default=1.0,
                    help="relgrad only: geometric decay of the step size so that the step at the LAST step "
                         "of the run is this fraction of the first. 1.0 = constant. It is given per RUN "
                         "because the per-step factor depends on the step count, which is only known here: "
                         "0.02 over 12 epochs x 5,860 steps/epoch = 70,320 steps is 0.99994437 per step.")
    ap.add_argument("--probe-grads", action="store_true",
                    help="CALIBRATION ONLY, no training: run ONE batch's forward/backward through the REAL "
                         "data path, mask and loss, print every tensor's size, gradient norm and the relgrad "
                         "step size that would match SGD at --lr, then exit. It produces no accuracy claim; "
                         "it exists because relgrad's per-entry movement is opt_lr/sqrt(n), so a step size "
                         "that suits a 5-parameter head would be ~1000x too small on a 3M-entry embedding.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default="run1")
    # RESUME: continue a run from a saved checkpoint instead of starting from scratch. Why this
    # exists: the trainer had no resume, so shutting the PC down mid-run threw away hours of GPU
    # work (measured: a 2.7 h attention run was about to be lost to a 21:40 shutdown).
    #   --resume <checkpoint.npz>   load its parameters into this run's model (shapes must match)
    #   --resume-ep N               the epoch that checkpoint IS; training continues at N+1
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-ep", type=int, default=None)
    ap.add_argument("--resume-any-corpus", action="store_true")
    # MID-EPOCH CHECKPOINT + EXACT MID-EPOCH RESUME (added 2026-10-23, self-wake). WHY: an epoch of the
    # FULL corpus under the gated/'ours' recipe measures 10.8-13.8 h (lens/runs/OURS-METHOD-COST.txt) and
    # this machine is stopped most nights, so epoch-boundary saving alone cannot produce ANY artifact from
    # that run -- not "less progress", nothing. Predictions P81a-f: lens/runs/MID-CKPT-RESUME-PREDICTIONS-P81.txt.
    # Both flags default OFF, and with them off this file writes no new file and behaves as it always did.
    ap.add_argument("--ckpt-seconds", type=float, default=0.0,
                    help="mid-epoch checkpoint + resume state every N seconds (0 = off, the default). "
                         "Writes {out}_mid.pkl (weights + step rule state + both gates' state + the epoch's "
                         "running totals + best-checkpoint memory) and {out}_ep{ep}_rngstart.json (the "
                         "generator state BEFORE that epoch's shuffle, which is what makes a resume exact "
                         "without storing a 135 MB index array).")
    ap.add_argument("--resume-mid", default=None,
                    help="resume from a mid-epoch checkpoint: 'auto' = {out}_mid.pkl, or an explicit path. "
                         "Refuses if the settings fingerprint or the training-window count differs from the "
                         "checkpoint's, so a resume cannot silently mix two experiments under one --out tag.")
    ap.add_argument("--max-windows", type=int, default=900000)
    ap.add_argument("--mtp", type=int, default=1, choices=[1, 2],
                    help="multi-token prediction: 2 adds a second head predicting the token two steps "
                         "ahead from the same concept vector. Published benefit is largest at or below "
                         "30M non-embedding parameters (arXiv:2404.19737) and our target model is capped "
                         "at 20M. Default 1 = the original behaviour, unchanged.")
    ap.add_argument("--train-turn-end", action="store_true",
                    help="ALSO train the token that starts the next human turn (the end-of-turn "
                         "signal). Without it the model is never taught to stop talking: measured "
                         "0/10 free generations handed the turn back. Off by default so every earlier "
                         "run stays reproducible.")
    ap.add_argument("--assistant-only", action="store_true",
                    help="mask the next-token loss to ASSISTANT tokens only. Measured problem this "
                         "addresses: the loss is currently computed over the whole window, so the "
                         "model spends capacity learning to re-predict the HUMAN's text as well as "
                         "its own replies. A chatbot is judged on its own turn, so training on the "
                         "user's words is wasted work. Implementation uses softmax_ce's existing "
                         "class_weights (weight 1 on assistant targets, 0 on human targets), so the "
                         "model only pays for predicting the assistant's own words (see loss "
                         "docstring); the loss scale stays comparable to the unmasked run.")
    ap.add_argument("--direct", action="store_true",
                    help="position-aware direct path from the token-level window to the answer (fixes the "
                         "measured bottleneck: the answer layer otherwise sees only the concept values)")
    ap.add_argument("--direct-mode", default="lowrank", choices=["lowrank", "tied"],
                    help="lowrank = the rank-16 path (measured NOT sufficient: val 28.06%% vs 28.30%% baseline, "
                         "talk still salad); tied = full-rank per-position mix read out through the model's own "
                         "embedding, which is the principled version")
    ap.add_argument("--direct-rank", type=int, default=16,
                    help="rank of the direct path; cost is W*D*rank + rank*V parameters")
    ap.add_argument("--attention", action="store_true",
                    help="causal self-attention before the concept layer (the architecture lever)")
    ap.add_argument("--drop-unk", action="store_true",
                    help="stop paying for <unk> targets: set the TRAINING weight to 0 at every target "
                         "position whose target token id is 0 (<unk>). Those positions are unanswerable by "
                         "construction (the true token was discarded by the vocabulary before training), "
                         "so the plain objective can only teach the model to EMIT <unk> -- measured on the "
                         "code corpus as the #1 most frequent target (lens/scripts/run_code_dropunk.sh) and "
                         "on this corpus at 6.888%% of held-out assistant targets with answer.b[<unk>] the "
                         "largest of 12,002 biases (P71f). Validation is NOT changed by this flag, so the "
                         "held-out columns stay comparable to runs without it. Needs --assistant-only (the "
                         "per-target weights only exist there) and refuses loudly without it. Off by "
                         "default, so every earlier run stays reproducible.")
    # --- SELF-CONTEXT / EXPOSURE-BIAS TRAINING (2026-10-25 self-wake; the one ship-path axis never trained) ---
    # WHY, with the measurements: every arm here is accurate on TEACHER-FORCED text and collapses after ONE of
    # its own tokens -- lens/runs/TALK-FAIR-finalbest-d256-repro.txt: step-8 free-running 3.9% against
    # teacher-forced 23.0%; the window-64 arm 1.6% against 22.3%; the ours arm 2.3% against 20.7%. Training has
    # always fed the model REAL corpus tokens as context (window = ids[Str : Str+window], target = the true next
    # token), so it has never once been asked to continue its OWN text. Context("window 64") is closed by
    # measurement (P95c/d), as are the decoder (P73/P74), the readout (P77), attention (P82) and the <unk>
    # objective (P89) -- this axis is what is left, and it is a TRAINER change, which is why it is built when no
    # arm is live (an arm that resumes mid-epoch would otherwise run under two versions of this file).
    ap.add_argument("--self-context-frac", type=float, default=0.0,
                    help="probability that a training step is a SELF-CONTEXT step: the window is rolled forward "
                         "by --self-context-rounds tokens built from the model's OWN greedy predictions, while "
                         "the TARGET stays the TRUE next token. 0.0 = OFF and bit-identical to this trainer "
                         "without the flag. Needs --mtp 1 and refuses loudly otherwise.")
    ap.add_argument("--self-context-rounds", type=int, default=4,
                    help="how many of the model's own tokens are rolled into the window per self-context step "
                         "(1..window-1). 1 = predict the true next token after your own first guess; larger = "
                         "your own errors compound before you are asked again.")
    # --- THE MEMORISATION RULE, PER STEP (PLAN item d2b, 2026-10-21) ----------------------------------
    # Until now this file carried only --max-gap: an EPOCH-level stop on (train accuracy - held-out
    # accuracy). That is a post-mortem at epoch granularity. LEARN-NOT-MEMORISE.md section 2a asks for the
    # rule to live in the STEP: measure the model's own probe before and after the candidate update and
    # REFUSE the step if it makes the held-out loss worse. The rule itself is the one lens/unified.py
    # already measured and calibrated (GapGate, EMA-smoothed, plus the calibration in
    # lens/scripts/gap_gate_calibration.py) - this flag wires THAT rule into the trainer that ships, with
    # the METHOD's own default (beta 0.5, tol 0.0, variant 'ema_band', exactly what train_unified()
    # builds when no gate is passed).
    ap.add_argument("--gates", choices=["off", "gap", "agreement", "gap+agreement"], default="off",
                    help="off = this file's behaviour before this flag existed, unchanged. gap = THE "
                         "MEMORISATION RULE FROM lens/unified.py, applied per STEP: the candidate update is "
                         "applied, the model's own held-out probe is re-scored, and a step that worsens the "
                         "smoothed probe loss is REVERTED. Every gate number is written per step to "
                         "<out>_gates.jsonl. agreement = THE SECOND DETECTOR (section 2b), applied per STEP "
                         "BEFORE the update: the batch is cut into --agree-groups sub-batches, the coherence "
                         "A = ||sum_g g_g|| / sum_g ||g_g|| is measured on the batch and on the model's own "
                         "held-out probe, and a step the detector refuses is SKIPPED (the weights do not "
                         "move). 'gap+agreement' runs both. The agreement numbers go to "
                         "<out>_agreement.jsonl so the gap gate's schema is untouched.")
    ap.add_argument("--gate-variant", default="ema_band",
                    choices=["ema_band", "divergence", "best_band", "gap_level", "rate_ratio",
                             "trend_band", "gap_trend", "off"],
                    help="which GapGate detector (see lens/unified.py for each one's measured record). "
                         "'off' is the CONTROL: it refuses nothing and must be digit-identical to --gates off. "
                         "'trend_band' is the near-miss RECOMMENDED to the owner and NOT applied anywhere "
                         "(lens/runs/GAP-GATE-DETECTOR4.txt: NOISE 0.787/0.833/0.277 against a >= 0.50 bar).")
    ap.add_argument("--gate-beta", type=float, default=0.5,
                    help="EMA smoothing of both losses (the method's default; a single mini-batch is far too "
                         "noisy to call memorisation on)")
    ap.add_argument("--gate-tol", type=float, default=0.0, help="GapGate tolerance (the method's default)")
    ap.add_argument("--gate-lag", type=int, default=25, help="lag for the lagged detectors (trend_band/gap_trend)")
    ap.add_argument("--probe-windows", type=int, default=256,
                    help="size of the model's OWN probe slice, drawn ONCE from the held-out split with "
                         "--probe-seed and never trained on. It is scored with the SAME loss the training "
                         "uses (assistant-masked when --assistant-only is on): scoring the probe against a "
                         "target the model was never asked to match is the bug that made teacher-mode "
                         "distillation learn neither the teacher nor the labels (2026-10-20).")
    ap.add_argument("--probe-seed", type=int, default=None, help="default: --seed")
    ap.add_argument("--gate-log-every", type=int, default=25,
                    help="print the gate numbers every N steps; the per-step JSONL is written every step")
    # ---- THE AGREEMENT GATE (LEARN-NOT-MEMORISE.md section 2b, PLAN item d2d) -------------------------
    ap.add_argument("--agree-variant", default="vs_probe", choices=["off", "level", "ema_band", "vs_probe"],
                    help="which AgreementGate detector (lens/unified.py). 'vs_probe' is the DEFAULT and the "
                         "only one with no human-chosen scale: refuse when the batch agrees with itself LESS "
                         "than the model's own held-out probe does. 'ema_band' is the direct analogue of the "
                         "gap gate's shipped detector (one noisy number against its own EMA). 'level' is a "
                         "FIXED threshold, kept so it can be measured rather than believed. 'off' is the "
                         "CONTROL: it refuses nothing and must be digit-identical to --gates off.")
    ap.add_argument("--agree-tol", type=float, default=0.0,
                    help="the detector's tolerance: level -> the absolute floor on A; ema_band -> how far "
                         "below its own EMA A may fall; vs_probe -> how far below the probe's A it may fall")
    ap.add_argument("--agree-beta", type=float, default=0.5, help="EMA smoothing of A for 'ema_band'")
    ap.add_argument("--agree-groups", type=int, default=8,
                    help="how many sub-batches the batch (and the probe slice) is cut into for the "
                         "coherence measurement. The independence floor is 1/sqrt(G): 0.354 at 8, 0.5 at 4.")
    ap.add_argument("--agree-probe-windows", type=int, default=64,
                    help="how many of the model's own probe windows the probe-side coherence is measured on "
                         "(the probe slice itself is drawn once with --probe-seed and never trained on)")
    ap.add_argument("--agree-log-every", type=int, default=25,
                    help="print the agreement numbers every N steps; the per-step JSONL is written every step")
    ap.add_argument("--method", choices=["ours", "standard"], default="standard",
                    help="ours = OUR one training method: our step rule (lens/opt.py relgrad, lr measured) + "
                         "our memorisation gate + our agreement gate (lens/unified.py), all ON, one switch. "
                         "standard = the old loop: plain SGD, gates off (the default, so nothing silently "
                         "changes for the existing runs and checkpoints)")
    a = ap.parse_args()

    # STOP-WITHOUT-LOSS (2026-10-26). Yahya leaves at no fixed time, so stopping has to be safe at ANY moment.
    # A checkpoint is already written every epoch; this handler writes an INTERRUPT checkpoint for the epoch in
    # flight and exits cleanly, so 'glassnet.sh off' cannot throw away half an epoch of steps.
    def _on_sigterm(signum, frame):
        try:
            log("")
            log(f"  SIGTERM: saving an interrupt checkpoint for epoch {globals().get('_cur_ep', 0)}")
            if "net" in globals():
                lens.save(net._net, os.path.join(OUT, f"{a.out}_interrupt_ep{globals().get('_cur_ep', 0)}.npz"))
            if "rng" in globals():
                try:
                    json.dump(rng.bit_generator.state, open(os.path.join(OUT, f"{a.out}_interrupt_rng.json"), "w"))
                except Exception:
                    pass
            log("  interrupt checkpoint saved, exiting cleanly")
        except Exception as e:
            log(f"  interrupt save FAILED: {e}")
        raise SystemExit(143)
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:
        pass

    # ---- OUR METHOD, AS ONE SWITCH (2026-10-23) --------------------------------------------------
    # Yahya's sequence, which this project had not been following: build OUR training method FIRST, then find
    # ITS best settings. The method existed but was never the thing doing the training: --opt defaulted to sgd
    # and --gates defaulted to off, so every run reported until now trained on the old loop with our method
    # switched off. The settings measured that way are settings for the old loop, which is exactly what he
    # warned would not transfer. This block makes one flag mean the whole method.
    method_note = "standard (the old loop: plain SGD, our gates OFF)"
    if getattr(a, "method", "standard") == "ours":
        a.opt = "relgrad"                    # our step rule, not plain SGD
        if a.opt_lr is None:
            # the measured match: the median relgrad lr that reproduces SGD at 0.4 in weight space
            # (lens/runs/OPT-GRAD-SCALE.txt). Not a taste value.
            a.opt_lr = 0.0441256
        a.gates = "gap+agreement"            # our memorisation rule + our agreement rule
        a.gate_variant = "ema_band"          # the method's own default detector
        a.agree_variant = "vs_probe"         # the method's own default
        method_note = ("OURS (relgrad step rule + gap gate + agreement gate; "
                       "lens/opt.py + lens/unified.py)")

    log("=" * 96)
    log("A MODEL THAT TALKS — licensed human dialogue, our own architecture")
    log(f"  METHOD: {method_note}")
    log("=" * 96)

    # --self-context-* validation (2026-10-25 self-wake): refuse loudly rather than run a recipe that is not
    # the experiment. IT SITS HERE, BEFORE THE CORPUS IS LOADED, on purpose: the mtp=2 branch below builds the
    # FULL uncapped window set (7-8 GB), so a refusal placed after it would cost the machine the exact memory
    # spike this project has already been OOM-killed by.
    if a.self_context_frac > 0.0:
        if int(getattr(a, "mtp", 1)) != 1:
            raise SystemExit("--self-context-frac requires --mtp 1: the mtp=2 second head targets "
                             "ids[Str+window+1] from the TRUE corpus, so a rolled-in window would mis-align "
                             "it. Refusing rather than mixing two experiments in one run.")
        if not (0.0 < a.self_context_frac <= 1.0):
            raise SystemExit("--self-context-frac must be in (0, 1] (0 disables the lever entirely)")
        if not (1 <= a.self_context_rounds <= a.window - 1):
            raise SystemExit(f"--self-context-rounds must be in 1..{a.window - 1} "
                             f"(got {a.self_context_rounds}); a window cannot hold more self-made tokens "
                             f"than it has positions minus one")
        log(f"  SELF-CONTEXT (exposure bias): {a.self_context_frac:.3f} of the steps train on a window whose "
            f"LAST {a.self_context_rounds} tokens are the model's OWN greedy predictions (the target is still "
            f"the TRUE next token), and the concept target stays the ORIGINAL window's (model.loss concept_ids).")
        log(f"  SELF-CONTEXT cost: ~{1.0 + a.self_context_frac * a.self_context_rounds:.2f}x the forward "
            f"passes per step (the roll-in forwards are inference-only). The step draw is made at the TOP of "
            f"the batch loop so a mid-epoch resume consumes the SAME random sequence as an uninterrupted run.")

    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    if int(getattr(a, "mtp", 1)) >= 2:
        # MTP needs each window's absolute start so the second head can target ids[start+window+1].
        Xtr, Ytr, Str = chat_data.make_windows(ids, bounds, split, a.window, which=0,
                                               return_index=True)
        Ytr2 = ids[Str + a.window + 1].astype(np.int64)
        log(f"  mtp={a.mtp}: second head targets built for {len(Ytr2):,} windows "
            f"(predicts the token two steps ahead from the same concept vector)")
    else:
        # pass the cap HERE: the old code built all 16,878,132 windows (7-8 GB, OOM-killed repeatedly) and
        # threw most away afterwards. Sampling at build time never materialises them.
        Xtr, Ytr, Str = chat_data.make_windows(ids, bounds, split, a.window, which=0,
                                               max_windows=max(a.max_windows * 2, 300000),
                                               seed=a.seed, return_index=True)
        Ytr2 = None
    Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, a.window, which=1, return_index=True)
    # NOTE: the cap on training windows is applied BELOW, after the assistant mask exists, so that
    # the mask can be subset with exactly the same indices. A prefix cap used to live here and it
    # silently trained on the first data source only (see the bug note below).

    # --- ASSISTANT-ONLY MASK -------------------------------------------------------------
    # Which TARGET positions belong to the assistant's turn? Built from the raw text, because the
    # token stream alone does not mark speaker turns: the tokens "assistant" and ":" appear as
    # ordinary tokens. So: re-tokenize the same conversations the same way, walk the token stream,
    # and flip a flag when an "assistant :" pair is seen. A window's target is assistant text if the
    # flag is ON at that token's position.
    #
    # This is done ONLY when --assistant-only is passed. Without it, nothing about the run changes.
    # --drop-unk (2026-10-24 self-wake) needs the PER-TARGET training weights, which are built only under
    # --assistant-only. Refuse loudly instead of silently training the unchanged recipe: a silent no-op
    # would produce a run that LOOKS like the experiment and is not one.
    if a.drop_unk and not a.assistant_only:
        raise SystemExit("--drop-unk requires --assistant-only (the per-target weights are built there); "
                         "refusing to run rather than silently doing nothing.")
    Wtr = None
    if a.assistant_only:
        vmap = {w: i for i, w in enumerate(vocab)}
        is_asst = np.zeros(len(ids), dtype=bool)
        # TURN-END TARGETS (added 2026-10-18). Measured problem: with plain assistant-only masking the
        # token that STARTS THE NEXT HUMAN TURN is a human target, so it gets weight 0 -- meaning the
        # model is never trained to emit the turn boundary, and a stop-control test measured 0/10
        # generations handing the turn back. This is the same bug two published projects document
        # (a chat terminator masked out of the loss, so "the only thing never learned is stopping").
        # With --train-turn-end those boundary tokens are trained like assistant text, because emitting
        # them is exactly how the model says "I am done, your turn".
        # float32 -> bool (2026-10-22 16:4x width fix): this array holds exactly 0/1 and is 1.18 GB at
        # 294M tokens as float32, 0.29 GB as bool. Only the TRAINING weights
        # need float, and they are converted explicitly where they are built (Wtr, Wpr); Wva is used as a
        # boolean selector (`Wva[idx] > 0`). Values are unchanged: memfix2 mask check, lens/runs/MEMFIX2-MASK.txt
        is_turnend = np.zeros(len(ids), dtype=bool)
        # walk token positions of the concatenated corpus, tracking speaker
        in_asst = False
        pos = 0
        for j in range(len(bounds) - 1):
            a0, b0 = bounds[j], bounds[j + 1]
            toks = [vocab[i] for i in ids[a0:b0]]
            k = 0
            while k < len(toks):
                # STANDARD CONTROL TOKENS (ChatML). Both formats are handled so every earlier corpus
                # still trains exactly as before: <|im_start|> role ... <|im_end|> (single dedicated
                # ids) and the older two-token "human :" / "assistant :" pairs.
                if toks[k].lower() == "<|im_start|>" and k + 1 < len(toks):
                    in_asst = toks[k + 1].lower() == "assistant"
                    k += 2
                    continue
                if toks[k].lower() == "<|im_end|>":
                    # this is the assistant's own stop token: the target the model must learn to emit
                    if in_asst and k > 0:
                        is_turnend[a0 + k] = True
                    k += 1
                    continue
                if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
                    in_asst = True
                    k += 2
                    continue
                if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
                    if in_asst and k > 0:
                        # the marker token itself is the end-of-turn signal the model must learn to emit
                        is_turnend[a0 + k] = True
                    in_asst = False
                    k += 2
                    continue
                if in_asst:
                    is_asst[a0 + k] = True
                k += 1
        # now weight each training window's target by whether that target token is assistant text
        # (astype, not copy: is_asst is bool now, and the LOSS needs float32 weights; 0/1 are exact in both)
        # MASKFIX 2026-10-22 18:0x — POSITION, NOT TOKEN ID (BUGS.md [BUG-MASK-TOKEN-ID]).
        # `Ytr` holds TOKEN IDS (chat_data: Y = ids_arr[starts + window]), so `is_asst[Ytr]` was reading
        # corpus positions 0..12,001 as if they were vocabulary indices: the weight became a property of
        # the TARGET TOKEN ID, not of whose turn the token is in, and "--assistant-only" never masked a
        # turn. Measured: by-token-id 51.0% of held-out windows vs by-position 66.9%, only 33.9% overlap
        # (lens/runs/MASK-INDEX-BUG.txt). The fix is the expression the comment above always described:
        # the TARGET'S OWN CORPUS POSITION. The legacy column is still computed and printed, labelled, so
        # no stored number changes meaning silently. Predictions: lens/runs/MASKFIX-PREDICTIONS.txt.
        Wtr_legacy = is_asst[Ytr].astype(np.float32)
        Wtr = is_asst[Str + a.window].astype(np.float32)
        frac = float(Wtr.mean()) if len(Wtr) else 0.0
        frac_legacy = float(Wtr_legacy.mean()) if len(Wtr_legacy) else 0.0
        log(f"  assistant-only (BY POSITION, the fixed mask): {int(Wtr.sum()):,}/{len(Wtr):,} training "
            f"targets are assistant text ({frac*100:.1f}%); human targets get weight 0")
        log(f"  assistant-only (BY TOKEN ID, LEGACY — what every run before 2026-10-22 18:0x used, kept "
            f"for comparison only): {int(Wtr_legacy.sum()):,}/{len(Wtr_legacy):,} ({frac_legacy*100:.1f}%)")
        if a.train_turn_end:
            te = is_turnend[Str + a.window] > 0.5
            te_legacy = is_turnend[Ytr] > 0.5
            Wtr[te] = 1.0
            log(f"  train-turn-end: {int(te.sum()):,} turn-boundary targets added to the loss "
                f"({te.mean()*100:.3f}% of training targets) -- without this the model is never taught "
                f"to stop talking, which is what the stop-control test measured at 0/10")
            log(f"  train-turn-end (LEGACY by token id): {int(te_legacy.sum()):,} "
                f"({te_legacy.mean()*100:.3f}%), listed only so the old count can be recognised")
            if int(te.sum()) == 0:
                log("  WARNING: train-turn-end was requested but no boundary targets were found; "
                    "check that the corpus actually contains 'human :' markers after assistant turns.")
        if frac == 0.0:
            log("  WARNING: no assistant targets found — the mask is empty, which would train on "
                "nothing. Check the speaker markers before trusting this run.")
    # MEMORISATION GATE (user's rule, 2026-10-18): the model must LEARN, never MEMORISE. A widening gap
    # between train and held-out accuracy is the signature of memorising -- measured today at 192 epochs:
    # train 43.19% vs honest 4.62%, a 38-point gap that looked like progress and was not. With --max-gap
    # the run STOPS when that gap opens, so training cannot quietly buy memorisation.
    # (the window subsample happens just above this line, after the assistant mask exists, so that
    #  the mask can be subset with the same indices)
    # --- TRAINING-WINDOW CAP ------------------------------------------------------------------
    # BUG FIX (2026-10-15): this used to be a PREFIX -- Xtr[:max_windows]. The corpora are
    # CONCATENATED BY SOURCE (hh first, then oasst1/oasst2/ultrachat/orca_math/...), so a prefix
    # silently trained on the first source(s) only, while the held-out VALIDATION spanned every
    # source. The measured symptom: the merged-corpus run drove train loss down to 0.69 while
    # clean-val accuracy sat at 10.61% (constant-token baseline 4.18%, uniform chance 0.01%) and val
    # perplexity climbed to 880,707 -- numbers that are implausible, which is exactly what exposed
    # the bug. Now: a SEEDED UNIFORM SUBSAMPLE across the whole corpus, so every data source is
    # represented. The choice is printed so no run can hide which windows it trained on.
    if len(Xtr) > a.max_windows:
        keep = np.random.default_rng(a.seed).choice(len(Xtr), size=a.max_windows, replace=False)
        keep.sort()
        lo_cov = float((keep < len(Xtr) * 0.25).mean())
        hi_cov = float((keep > len(Xtr) * 0.75).mean())
        Xtr, Ytr = Xtr[keep], Ytr[keep]
        if Ytr2 is not None:
            Ytr2 = Ytr2[keep]
        Wtr = Wtr[keep] if Wtr is not None else None
        log(f"  window cap: {len(keep):,} windows chosen UNIFORMLY AT RANDOM across the whole corpus "
            f"(seeded {a.seed}); {lo_cov*100:.1f}% come from the first quarter of the corpus and "
            f"{hi_cov*100:.1f}% from the last quarter (a prefix cap would read 100%/0%)")
    # --- DROP-UNK (2026-10-24 self-wake, candidate (ii) of P78f/P82e) ----------------------------------
    # Moved HERE, AFTER the window cap, in the same turn it was written (BUGS.md 2026-10-24). Reason, measured:
    # the first version sat in the assistant-mask section ABOVE the cap, so on a capped run it counted over the
    # BUILT window set and then the cap subset the weights afterwards: the d=64 pre-flight printed 12,549
    # dropped of 300,000 built while the arm actually trained on 60,000, and a reader comparing that count
    # against the training set it describes would have been comparing two different sets. The WEIGHTS are
    # identical either way (zeroing then subsetting == subsetting then zeroing), so no number of any run
    # changes; only the set the PRINT describes. Verified digit for digit: see
    # lens/runs/UNK-DROP-PREFLIGHT.txt (same epoch line with the block above and below the cap).
    # Every target position whose TARGET TOKEN ID is 0 (<unk>) is removed from the loss. Those positions are
    # unanswerable by construction: the true token was discarded by the vocabulary before training began, so
    # the only thing the objective can teach there is to EMIT <unk>. Measured on the code corpus as the #1 most
    # frequent target (lens/scripts/run_code_dropunk.sh); on THIS corpus 6.888% of held-out assistant targets
    # are id 0 and answer.b[<unk>] is the largest of 12,002 biases (P71f, lens/runs/PADFREE-TALK.txt line 4).
    # Applied to the TRAINING weights only -- validation reads is_asst by position and is untouched, so every
    # held-out column stays comparable to runs without this flag.
    # Predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89a-c + the two addenda).
    if a.drop_unk:
        if Wtr is None:
            raise SystemExit("--drop-unk needs --assistant-only (the per-target weights are built there); "
                             "refusing to run rather than silently doing nothing.")
        w_was = float(Wtr.sum())
        unk = (Ytr == 0)
        dropped = int(((Wtr > 0) & unk).sum())
        if dropped == 0:
            raise SystemExit("--drop-unk found 0 <unk> targets among the weighted training targets; "
                             "that is not the experiment it claims to be. Refusing to run.")
        Wtr[unk] = 0.0
        log(f"  drop-unk: {dropped:,} training targets whose own token is <unk> (id 0) removed from the "
            f"loss -- {dropped/max(int(Wtr.size),1)*100:.3f}% of the {int(Wtr.size):,} training targets this "
            f"run actually trains on, {dropped/max(w_was,1.0)*100:.3f}% of the WEIGHTED (assistant) ones; "
            f"weighted targets {w_was:,.0f} -> {float(Wtr.sum()):,.0f}")
        _wva = is_asst[np.asarray(Sva) + a.window]
        _wat = int(_wva.sum())
        _unkva = int(((Yva == 0) & _wva).sum())
        log(f"  drop-unk: held-out assistant targets that are <unk>: {_unkva:,} of {_wat:,} "
            f"({_unkva/max(_wat,1)*100:.3f}%) -- VALIDATION IS NOT CHANGED by this flag, so the "
            f"held-out column stays comparable to runs without it")
    log(f"  windows {len(Xtr):,} train / {len(Xva):,} val   vocab {len(vocab):,}   window {a.window}")

    # --- THE MODEL'S OWN PROBE, AND THE PER-STEP MEMORISATION RULE (PLAN item d2b) ---------------------
    # The probe is drawn ONCE, here, from the HELD-OUT split, with a recorded seed, and it is never trained
    # on. Both the draw and its size are printed, because "the model measures itself" is only meaningful if
    # you can see what it measures itself on. Nothing below runs unless --gates is not off, so every earlier
    # run of this file is bit-identical (checked digit for digit by lens/scripts/gate_smoke.sh against
    # lens/runs/_tmp/train_chat_prefix_gates.py).
    gate = None
    agate = None
    Xpr = Ypr = Wpr = None
    gate_path = None
    agate_path = None
    gate_stats = {"checked": 0, "refused": 0, "accepted": 0, "probe_disjoint_overlap": None,
                  "probe_first": None, "probe_last": None}
    astats = {"checked": 0, "refused": 0, "accepted": 0}
    use_gap = a.gates in ("gap", "gap+agreement")
    use_agree = a.gates in ("agreement", "gap+agreement")
    if a.gates != "off":
        if use_gap:
            from lens.unified import GapGate
            gate = GapGate(beta=a.gate_beta, tol=a.gate_tol, variant=a.gate_variant, lag=a.gate_lag)
        if use_agree:
            from lens.unified import AgreementGate
            agate = AgreementGate(variant=a.agree_variant, tol=a.agree_tol, beta=a.agree_beta,
                                  groups=a.agree_groups)
        pseed = a.seed if a.probe_seed is None else a.probe_seed
        npr = int(min(a.probe_windows, len(Xva)))
        prng = np.random.default_rng(pseed)
        pi = np.sort(prng.choice(len(Xva), size=npr, replace=False))
        Xpr, Ypr = Xva[pi], Yva[pi]
        # MASKFIX: the probe's weights were by TOKEN ID too; same fix, same reason (BUGS.md [BUG-MASK-TOKEN-ID]).
        Wpr = np.asarray(is_asst[Sva[pi] + a.window], dtype=np.float32) if a.assistant_only else None
        log(f"  THE MODEL'S OWN PROBE: {npr:,} of {len(Xva):,} held-out windows, drawn ONCE with seed {pseed}, "
            f"never trained on; scored with the SAME loss the training uses")
        if use_gap:
            log(f"  MEMORISATION RULE (lens/unified.py, LEARN-NOT-MEMORISE.md 2a): GapGate(variant={a.gate_variant!r}, "
                f"beta={a.gate_beta:g}, tol={a.gate_tol:g}, lag={a.gate_lag}); a refused step is REVERTED")
        if use_agree:
            log(f"  AGREEMENT RULE (lens/unified.py, LEARN-NOT-MEMORISE.md 2b): "
                f"AgreementGate(variant={a.agree_variant!r}, tol={a.agree_tol:g}, beta={a.agree_beta:g}, "
                f"groups={a.agree_groups}); a refused step is SKIPPED (the weights do not move)")
            log(f"  the coherence number A = ||sum_g g_g|| / sum_g ||g_g|| over {a.agree_groups} sub-batches; "
                f"the independence floor at {a.agree_groups} groups is {1.0 / np.sqrt(a.agree_groups):.4f}")
            log(f"  A is measured on the batch AND on the first "
                f"{int(min(a.agree_probe_windows, len(Xpr))):,} of the probe windows; the update gradient is "
                f"UNCHANGED by this rule (the full-batch gradient is snapshotted and restored around the "
                f"agreement pass)")
        # The probe must be disjoint from the training windows. Checked exactly on window CONTENT (the
        # window's tokens plus its target token), because the drawn indices alone do not prove it. Above
        # 200,000 train windows the walk is skipped and the skip is PRINTED with the count - a stated limit,
        # never a silent pass.
        if len(Xtr) <= 200000:
            def _wset(A, y):
                return {hash((bytes(np.ascontiguousarray(A[i]).data), int(y[i]))) for i in range(len(A))}
            overlap = len(_wset(Xpr, Ypr) & _wset(Xtr, Ytr))
            gate_stats["probe_disjoint_overlap"] = int(overlap)
            log(f"  probe disjointness: {overlap} of {npr:,} probe windows also appear in the train set "
                f"(0 = disjoint, checked on window content, not on indices)")
        else:
            log(f"  probe disjointness: SKIPPED - {len(Xtr):,} train windows is above the 200,000 limit for "
                f"the exact content check (stated, not passed silently)")
        if use_gap:
            gate_path = os.path.join(OUT, f"{a.out}_gates.jsonl")
            log(f"  per-step gate log: {gate_path}")
        if use_agree:
            agate_path = os.path.join(OUT, f"{a.out}_agreement.jsonl")
            log(f"  per-step agreement log: {agate_path}")

    # concept support measured on TRAIN tokens only
    train_mask = np.zeros(len(ids), dtype=bool)
    for j in range(len(bounds) - 1):
        if split[j] == 0:
            train_mask[bounds[j]:bounds[j + 1]] = True
    # the concept set: narrow (9, the original) or wide (~48, the measured fix for the bottleneck)
    cs_mod = "chat_concepts_wide" if os.environ.get("TC_WIDE", "1") == "1" else "chat_concepts"
    import importlib
    cmod = importlib.import_module(cs_mod)
    cs, spec = cmod.build(vocab, ids, train_mask, windows=Xtr)
    log(f"  concept module: {cs_mod}  ({spec.get('n_defined', '?')} defined)")
    log(f"  concepts: {spec['n_kept']} kept, {spec['n_dropped']} dropped (floor {spec['min_support']})")
    for c in spec["kept"]:
        log(f"     {c['name']:<22} support {c['support']:.4f}  {c['n_tokens']} tokens")
    for c in spec["dropped"]:
        log(f"     DROPPED {c['name']:<16} {c['reason']}")
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in cs]

    net = ChatNet(len(vocab), concepts, d=a.d, window=a.window, seed=a.seed,
                  lam_concept=a.lam_concept, lam_act=a.lam_act, device=a.device,
                  attention=a.attention, mtp=a.mtp, direct=a.direct, direct_rank=a.direct_rank, direct_mode=a.direct_mode)

    # T10-2026-10-11: concept-strength boost (ONE change from R5). The strict audit named
    # is_a_greeting / is_formal as weak: |own| vs |best-other| margin 0.09 (needs >0.10).
    # Their low corpus support starves the weighted-MSE concept loss, so the neurons never
    # track their formulas hard enough. Legacy: per-concept x3 on their pos_weight — the
    # neuron must work 3x harder to satisfy its own label. Binary bar pre-registered:
    # strict 43/43 AND val_acc >= 30.88.
    boost = os.environ.get("TALK_BOOST", "")
    for part in filter(None, boost.split(",")):
        name, mult = part.rsplit(":", 1)
        j = next(i for i, c in enumerate(concepts) if c.name == name)
        net._net.pos_weight[j] *= float(mult)
        log(f"     BOOST concept {name} pos_weight x{mult}")
    log(f"  model: {net.n_params():,} params  (d={a.d}, K={net.K})")
    tpp = len(Xtr) / max(net.n_params(), 1)
    log(f"  tokens per parameter: {tpp:.2f}  "
        f"({'well trained' if tpp > 10 else 'adequate' if tpp > 4 else 'UNDER-TRAINED'})")

    rng = np.random.default_rng(a.seed)
    start_ep = 1
    if a.resume:
        if a.resume_ep is None:
            raise SystemExit("--resume needs --resume-ep N (the epoch the checkpoint IS)")
        run_log = os.path.join(OUT, f"{a.out}_log.json")
        if os.path.exists(run_log):
            prev = json.load(open(run_log)).get("args", {})
            prev_dir, cur_dir = prev.get("data_dir"), os.environ.get("CHAT_DATA_DIR", "(default)")
            if prev_dir and prev_dir != cur_dir and not a.resume_any_corpus:
                raise SystemExit(
                    f"RESUME REFUSED: '{a.out}' was trained on corpus {prev_dir}, but this command "
                    f"points at {cur_dir}. Resuming across corpora would keep the run name while "
                    f"changing the data underneath it. Pass --resume-any-corpus only if that is "
                    f"genuinely intended.")
        z = np.load(a.resume, allow_pickle=True)
        loaded, bad = 0, []
        for pt in net._net.parameters():
            if pt.name not in z.files:
                bad.append((pt.name, "missing from the checkpoint"))
                continue
            arr = z[pt.name]
            if tuple(arr.shape) != tuple(pt.data.shape):
                bad.append((pt.name, f"shape {arr.shape} vs model {pt.data.shape}"))
                continue
            pt.data = B.to_dev(np.ascontiguousarray(arr))
            loaded += 1
        if bad:
            raise SystemExit(f"RESUME REFUSED: {bad}")
        start_ep = a.resume_ep + 1
        log(f"  RESUME: loaded {loaded} parameter tensors from {os.path.basename(a.resume)} "
            f"(epoch {a.resume_ep}) -> continuing at epoch {start_ep} of {a.epochs}")
        # the shuffle order decides which batches are seen; restoring the RNG state makes a resumed
        # run BIT-IDENTICAL to an uninterrupted one instead of merely 'in the same ballpark'
        rng_side = os.path.join(OUT, f"{a.out}_ep{a.resume_ep}_rng.json")
        if os.path.exists(rng_side):
            rng.bit_generator.state = json.load(open(rng_side))
            log(f"  RESUME: shuffle RNG state restored from {os.path.basename(rng_side)} "
                f"(exact continuation)")
        else:
            log("  RESUME: no RNG sidecar for this checkpoint (it predates RNG saving), so the "
                "shuffle order for the remaining epochs differs from an uninterrupted run. The "
                "parameters are exact; the batch ORDER is not. Stated, not hidden.")
    # ---- MID-EPOCH CHECKPOINT / RESUME (--ckpt-seconds, --resume-mid) ------------------------------
    # Added 2026-10-23 by the self-wake; predictions P81a-f in lens/runs/MID-CKPT-RESUME-PREDICTIONS-P81.txt,
    # measured cost in lens/runs/OURS-METHOD-COST.txt. NOTHING above this line changed: with --ckpt-seconds 0
    # and no --resume-mid, no new file is written and this file behaves exactly as it did.
    import hashlib
    _cfg = {k: v for k, v in vars(a).items() if k not in ("resume_mid", "ckpt_seconds")}
    _cfg["data_dir"] = os.environ.get("CHAT_DATA_DIR", "(default)")
    cfg_hash = hashlib.sha1(json.dumps(_cfg, sort_keys=True, default=str).encode()).hexdigest()
    mid_state = None
    mid_skip_step = 0
    if a.resume_mid:
        mid_pkl = (os.path.join(OUT, f"{a.out}_mid.pkl") if a.resume_mid == "auto" else a.resume_mid)
        if not os.path.exists(mid_pkl):
            raise SystemExit(f"RESUME-MID REFUSED: no mid-epoch checkpoint at {mid_pkl}")
        with open(mid_pkl, "rb") as fh:
            mid_state = pickle.load(fh)
        if int(mid_state.get("version", 0)) != 1:
            raise SystemExit(f"RESUME-MID REFUSED: {mid_pkl} is version {mid_state.get('version')}; this "
                             f"trainer writes version 1")
        if mid_state.get("cfg_hash") != cfg_hash:
            raise SystemExit("RESUME-MID REFUSED: the settings on this command line are not the settings the "
                             "checkpoint was written with (fingerprints differ). Resuming would keep the "
                             "run's name while changing the experiment underneath it.")
        if int(mid_state.get("train_windows", -1)) != len(Xtr):
            raise SystemExit(f"RESUME-MID REFUSED: that checkpoint trained on "
                             f"{mid_state.get('train_windows'):,} windows, this command builds {len(Xtr):,}")
        miss, loaded = [], 0
        for pt in net._net.parameters():
            arr = mid_state["params"].get(pt.name)
            if arr is None or tuple(arr.shape) != tuple(pt.data.shape):
                miss.append((pt.name, None if arr is None else tuple(arr.shape)))
                continue
            pt.data = B.to_dev(np.ascontiguousarray(arr))
            loaded += 1
        if miss:
            raise SystemExit(f"RESUME-MID REFUSED: tensors missing or wrong shape: {miss}")
        start_ep = int(mid_state["epoch"])
        mid_skip_step = int(mid_state["step"])
        log(f"  RESUME-MID: {loaded} tensors + full step state loaded from {os.path.basename(mid_pkl)} "
            f"(written {time.strftime('%F %H:%M:%S', time.localtime(mid_state.get('written_at', time.time())))})")
        log(f"    continuing EPOCH {start_ep} at step {mid_skip_step:,} of "
            f"{int(np.ceil(len(Xtr) / a.batch)):,}; the epoch's shuffle order is regenerated from the "
            f"epoch-start RNG sidecar, so this is an EXACT continuation, not an approximation")
    # gates' state: restored HERE rather than next to the gate constructors, because the gates are built
    # before the model and the checkpoint cannot be read until the model (and so the parameter shapes) exists
    if mid_state is not None:
        if gate is not None and mid_state.get("gate"):
            st = mid_state["gate"]
            gate.ema_train = st.get("ema_train")
            gate.ema_held = st.get("ema_held")
            gate.best_held = st.get("best_held")
            gate.hist_t = list(st.get("hist_t") or [])
            gate.hist_h = list(st.get("hist_h") or [])
            gate.checked = int(st.get("checked", 0))
            gate.refused = int(st.get("refused", 0))
            gate.gap_first = st.get("gap_first")
            gate.gap_last = st.get("gap_last")
        if agate is not None and mid_state.get("agate"):
            st = mid_state["agate"]
            agate.ema = st.get("ema")
            agate.a_first = st.get("a_first")
            agate.a_last = st.get("a_last")
            agate.checked = int(st.get("checked", 0))
            agate.refused = int(st.get("refused", 0))
        log(f"  RESUME-MID: gates restored (gap EMAs train {gate.ema_train if gate else None}, held "
            f"{gate.ema_held if gate else None}; agreement EMA {agate.ema if agate else None}) - a "
            f"resumed run decides its next step from the same smoothed numbers the interrupted one left")
    # --- THE STEP RULE (--opt), SET UP HERE ------------------------------------------------------
    # This file did plain SGD by hand until now (PLAN.txt line 32-33, FROM-SCRATCH.md GAP 2):
    #   p.data = p.data - lr_now * p.grad
    # --opt sgd is exactly that rule, bit-identical (the smoke regression checks it digit for digit
    # against lens/runs/_tmp/train_chat_prefix.py, the pre-change copy). --opt relgrad is OUR rule:
    # one step per batch, per tensor, of norm exactly opt_lr in weight space. The decay is expressed
    # PER RUN (--opt-decay-final) because the per-step factor depends on the step count, which is only
    # known here; a fixed per-step factor of 0.999 would collapse the step to 0.2% of its value by the
    # second epoch of this run (0.999**70,320 ~ 1e-31), which is not the measured setting.
    opt_lr = float(a.opt_lr if a.opt_lr is not None else a.lr)
    opt = None
    opt_decay_step = 1.0
    if a.opt == "relgrad":
        steps_per_ep = int(np.ceil(len(Xtr) / a.batch))
        total_steps = max(steps_per_ep * max(a.epochs - start_ep + 1, 1), 1)
        if not (0.0 < a.opt_decay_final <= 1.0):
            raise SystemExit("--opt-decay-final must be in (0, 1]")
        if a.opt_decay_final != 1.0:
            opt_decay_step = float(a.opt_decay_final) ** (1.0 / total_steps)
        opt = RelativeGrad(net.parameters(), lr=opt_lr, mode="relgrad",
                           momentum=a.opt_momentum, decay=opt_decay_step)
        log(f"  STEP RULE: relgrad (OUR rule, lens/opt.py)   opt_lr {opt_lr:g}  "
            f"momentum {a.opt_momentum:g}")
        log(f"    the update of every tensor has norm exactly {opt_lr:g} in weight space; --lr "
            f"({a.lr:g}) is NOT the step size for this rule and is not used")
        log(f"    decay: step at the last of {total_steps:,} steps ({steps_per_ep:,}/epoch) = "
            f"{a.opt_decay_final:g} x the first -> per-step factor {opt_decay_step:.8f}")
        if a.lr_decay > 0:
            log(f"    NOTE: --lr-decay {a.lr_decay:g} applies to --opt sgd only and is IGNORED for this "
                f"run; the relgrad schedule is --opt-decay-final. Stated, not silently dropped.")
    else:
        log(f"  STEP RULE: plain SGD (this file's own rule, unchanged)   lr {a.lr:g}"
            + (f" decaying linearly to {a.lr*(1.0-a.lr_decay):g} at the last epoch"
               if a.lr_decay > 0 else " (constant)"))

    if mid_state is not None and opt is not None and mid_state.get("opt"):
        st = mid_state["opt"]
        opt.steps = int(st["steps"])
        opt.lr = float(st.get("lr", opt.lr))
        opt.decay = float(st.get("decay", opt.decay))
        opt._ema = {k: np.asarray(v) for k, v in (st.get("ema") or {}).items()}
        opt._mom = {k: np.asarray(v) for k, v in (st.get("mom") or {}).items()}
        log(f"  RESUME-MID: step rule restored (mode {opt.mode!r}, step counter {opt.steps:,}, lr "
            f"{opt.lr:g}, per-step decay {opt.decay:.8f} - the ORIGINAL schedule's factor, not one "
            f"recomputed from the shortened run that is left)")
    if a.probe_grads:
        # CALIBRATION ONLY -- no accuracy claim comes out of this and it does not train.
        idx = rng.permutation(len(Xtr))[:a.batch]
        for p in net.parameters():
            p.grad = None
        total, _parts = net.loss(Xtr[idx], Ytr[idx],
                                 weights=(Wtr[idx] if Wtr is not None else None),
                                 y2=(Ytr2[idx] if Ytr2 is not None else None))
        total.backward()
        log("")
        log("  GRADIENT-SCALE PROBE (calibration only, NO accuracy claim, no training run):")
        log(f"  one batch of {len(idx):,} windows through the real data path, mask and loss; "
            f"loss {float(B.to_cpu(total.data)):.4f}")
        log(f"  {'tensor':<26} {'n':>10} {'||g||':>12} {'lr*||g|| (relgrad lr matched to SGD)':>36}")
        matched = []
        for p in net.parameters():
            if p.grad is None:
                continue
            g = np.asarray(B.to_cpu(p.grad), dtype=np.float64)
            nrm = float(np.linalg.norm(g))
            matched.append(a.lr * nrm)
            log(f"  {p.name:<26} {g.size:>10,} {nrm:>12.4e} {a.lr*nrm:>36.6g}")
        if matched:
            med = float(np.median(matched))
            log(f"  median relgrad lr that matches SGD@{a.lr:g}: {med:.6g}   "
                f"(mean {float(np.mean(matched)):.6g}, min {min(matched):.6g}, max {max(matched):.6g})")
            log(f"  -> the swept grid must straddle {med:.6g}. A grid centred on 0.02 would move every")
            log(f"     entry {med/0.02:.1f}x too little, which would read as 'our optimiser does not learn'.")
        log("  PROBE DONE - nothing was trained, no result of any kind was produced.")
        return

    hist = []
    gstep = 0
    astep = 0
    best_val, best_ep = None, None   # BEST-VAL TRACKING (added 2026-10-21)
    last_ckpt = time.perf_counter()
    steps_this_ep = int(np.ceil(len(Xtr) / a.batch))
    if mid_state is not None:
        gstep = int(mid_state.get("gstep", 0))
        astep = int(mid_state.get("astep", 0))
        for k, v in (mid_state.get("gate_stats") or {}).items():
            if k in gate_stats:
                gate_stats[k] = v
        for k, v in (mid_state.get("astats") or {}).items():
            if k in astats:
                astats[k] = v
        best_val, best_ep = mid_state.get("best_val"), mid_state.get("best_ep")
        log(f"  RESUME-MID: counters restored (gap gate {gate_stats['checked']:,} checks, agreement gate "
            f"{astats['checked']:,}); best held-out before the interruption: "
            f"{(f'{best_val*100:.2f}% at epoch {best_ep}') if best_val is not None else 'none yet'} - a "
            f"resumed run will NOT overwrite a better earlier *_best.npz by accident")
    # ---- SELF-CONTEXT (exposure bias) STATE (2026-10-25 self-wake; predictions P98) --------------------
    # A SEPARATE generator from the epoch-shuffle `rng`: turning this lever on must not change the shuffle
    # order, which is exactly what a mid-epoch resume regenerates from a saved sidecar. Seeded from --seed, so
    # the step pattern is reproducible and belongs to the run rather than to the machine.
    sctx_rng = np.random.default_rng(int(a.seed) + 7777)
    sctx_steps = 0
    sctx_hit = 0
    sctx_n = 0
    sctx_draw = 0.0
    for ep in range(start_ep, a.epochs + 1):
        t0 = time.perf_counter()
        resuming_this_ep = mid_state is not None and ep == int(mid_state["epoch"])
        if a.ckpt_seconds > 0 and not resuming_this_ep:
            # the epoch's shuffle order comes from the generator exactly as it stands right now, so saving
            # its state BEFORE the draw is what lets a mid-epoch resume regenerate the SAME order: no index
            # array on disk, and the order is exact rather than re-drawn.
            with open(os.path.join(OUT, f"{a.out}_ep{ep}_rngstart.json"), "w") as fh:
                json.dump(rng.bit_generator.state, fh)
        elif resuming_this_ep:
            side = os.path.join(OUT, f"{a.out}_ep{ep}_rngstart.json")
            if not os.path.exists(side):
                raise SystemExit(f"RESUME-MID REFUSED: {side} is missing, so this epoch's shuffle order "
                                 f"cannot be regenerated. Without it a resume would silently re-order the "
                                 f"data and stop being the run it claims to be.")
            rng.bit_generator.state = json.load(open(side))
            log(f"  RESUME-MID: epoch {ep} shuffle order regenerated from {os.path.basename(side)} "
                f"(the same windows in the same order as the uninterrupted run)")
        order = rng.permutation(len(Xtr))
        tot, ctot, nb, corr, seen = 0.0, 0.0, 0, 0, 0
        skip_to = 0
        if resuming_this_ep:
            skip_to = int(mid_skip_step) * a.batch
            st = mid_state.get("acc") or {}
            tot = float(st.get("tot", 0.0))
            ctot = float(st.get("ctot", 0.0))
            nb = int(st.get("nb", 0))
            corr = int(st.get("corr", 0))
            seen = int(st.get("seen", 0))
            last_ckpt = time.perf_counter()
            log(f"  RESUME-MID: EPOCH {ep} continues at window {skip_to:,} of {len(Xtr):,} "
                f"({int(mid_skip_step):,} of {steps_this_ep:,} steps already done); this epoch's train "
                f"numbers in this log INCLUDE them ({nb:,} batches, mean loss so far {tot / max(nb, 1):.4f})")
        # learning-rate decay: the second-ranked lever from the optimisation diagnostic. lr was
        # constant 0.4 in run 1 and perplexity was still falling at the last epoch, so a decay
        # schedule is tested as its own change rather than bundled with the concept widening.
        lr_now = a.lr * (1.0 - a.lr_decay * (ep - 1) / max(a.epochs - 1, 1)) if a.lr_decay > 0 \
            else a.lr
        for s in range(0, len(Xtr), a.batch):
            s_off = s               # TRAP, MEASURED 2026-10-23 (mid-ckpt smoke / BUGS.md): the agreement
                                    # pass below runs `for p, s in zip(net.parameters(), gsnap)` and CLOBBERS
                                    # this loop's `s` with a gradient ARRAY before the body ends. Anything
                                    # below that needs the window offset must read s_off, never `s`.
            if a.self_context_frac > 0.0:
                # DRAWN BEFORE the skip_to check: a resumed epoch must consume the same random sequence an
                # uninterrupted one would, so the self-context step PATTERN is part of the run, not of the
                # interruption. With the flag OFF nothing is drawn and nothing below changes.
                sctx_draw = float(sctx_rng.random())
            if skip_to and s < skip_to:
                continue            # the part of a resumed epoch that was already trained; `order` is the
                                    # SAME array the interrupted attempt used, so this is a jump, not a redo
            idx = order[s:s + a.batch]
            for p in net.parameters():
                p.grad = None
            # ---- SELF-CONTEXT STEP (exposure bias): train on the model's OWN text -------------------------
            # The window is slid left by one and the model's own greedy next token is appended, `rounds` times,
            # while the TARGET stays the TRUE next token. So the model is asked, at last, to predict real
            # English from a context that contains its own errors -- the measured failure is exactly that it
            # does this accurately only when the context is real (P65c: step-8 free-running 0.4-3.9%).
            Xstep = Xtr[idx]
            _selfctx = False
            if a.self_context_frac > 0.0 and sctx_draw < a.self_context_frac:
                Xs = np.array(Xtr[idx], copy=True)
                _yhat1 = None
                for _r in range(int(a.self_context_rounds)):
                    _lg = np.asarray(net.forward_arrays(Xs)["logits"])   # inference only: no graph, no grad
                    _yh = _lg.argmax(axis=1).astype(Xs.dtype)
                    if _r == 0:
                        _yhat1 = _yh
                    Xs = np.concatenate([Xs[:, 1:], _yh[:, None]], axis=1)
                sctx_steps += 1
                if _yhat1 is not None:
                    # round-1 own-token agreement with the TRUE next token = the model's own error rate at step
                    # 1 of free running, on TRAIN windows. Printed per epoch; it is a measurement of the
                    # exposure this arm is built to attack, not a quality score.
                    sctx_hit += int((_yhat1 == Ytr[idx]).sum())
                    sctx_n += int(len(_yhat1))
                Xstep = Xs
                _selfctx = True
            # the library's loss already contains both terms: next-token CE plus the balanced
            # concept BCE on the named concepts. Reimplementing it here is what broke the first
            # version three times over.
            total, parts = net.loss(Xstep, Ytr[idx],
                                    weights=(Wtr[idx] if Wtr is not None else None),
                                    y2=(Ytr2[idx] if Ytr2 is not None else None),
                                    concept_ids=(Xtr[idx] if _selfctx else None))
            total.backward()
            # ---- THE AGREEMENT RULE (LEARN-NOT-MEMORISE.md 2b, PLAN item d2d) --------------------------
            # Measured on the batch gradient BEFORE the step is taken, so a refused step needs no revert:
            # the weights simply do not move. The agreement pass re-uses p.grad (the library's backward
            # accumulates into it), so the FULL-BATCH gradient this step will use is snapshotted here and
            # restored immediately after - the update path below is the one it always was, unchanged.
            agree_refuse = False
            if agate is not None:
                gsnap = [None if p.grad is None else np.array(B.to_cpu(p.grad)) for p in net.parameters()]
                a_train, acc_tr, gused = agreement_coherence(net, Xstep, Ytr[idx],
                                                             (Wtr[idx] if Wtr is not None else None),
                                                             (Ytr2[idx] if Ytr2 is not None else None),
                                                             groups=a.agree_groups)
                nprA = int(min(a.agree_probe_windows, len(Xpr)))
                a_probe = (agreement_coherence(net, Xpr[:nprA], Ypr[:nprA],
                                               (Wpr[:nprA] if Wpr is not None else None), None,
                                               groups=a.agree_groups)[0] if nprA > 0 else None)
                for p, s in zip(net.parameters(), gsnap):
                    p.grad = None if s is None else B.to_dev(s)
                # P60m (MEASURED, NOT USED FOR THE UPDATE): what is the SUM of the sub-batch gradients
                # worth as a batch gradient? Measured on the FIRST step only, because the answer is a
                # property of the loss and the split, not of the training. If it matched, a future version
                # could have the agreement number with no extra pass at all.
                grel = None
                if astep == 0 and acc_tr is not None and gused > 0:
                    num = 0.0
                    for i, p in enumerate(net.parameters()):
                        ref = np.asarray(gsnap[i], dtype=np.float64).reshape(-1) if gsnap[i] is not None \
                            else np.zeros(p.data.size, dtype=np.float64)
                        got = (acc_tr[i] / float(gused)) if acc_tr[i] is not None \
                            else np.zeros(p.data.size, dtype=np.float64)
                        d = got - ref
                        num += float(np.dot(d, d))
                    dref = 0.0
                    for i, p in enumerate(net.parameters()):
                        ref = np.asarray(gsnap[i], dtype=np.float64).reshape(-1) if gsnap[i] is not None \
                            else np.zeros(p.data.size, dtype=np.float64)
                        dref += float(np.dot(ref, ref))
                    grel = float(np.sqrt(num) / np.sqrt(dref)) if dref > 0 else None
                    log(f"    P60m: ||sum_g g_g / G - full-batch grad|| / ||full-batch grad|| = "
                        f"{grel:.6e}  ({gused} sub-batches, MEASURED, not used for the update)")
                agree_refuse, cand_a = agate.evaluate(a_train, a_probe)
                astats["checked"] += 1
                astep += 1
                if agree_refuse:
                    astats["refused"] += 1
                else:
                    astats["accepted"] += 1
                    agate.commit(cand_a)
                with _open_log(agate_path) as ah:
                    ah.write(json.dumps({"step": astep, "epoch": ep, "a_train": float(a_train),
                                         "a_probe": (None if a_probe is None else float(a_probe)),
                                         "ema_a": (None if agate.ema is None else float(agate.ema)),
                                         "batch_ce": float(B.to_cpu(parts.get("ce", total.data))),
                                         "variant": agate.variant, "tol": agate.tol,
                                         "groups_used": int(gused),
                                         "grad_group_sum_rel": grel,
                                         "refused": bool(agree_refuse),
                                         "accepted": bool(not agree_refuse)}) + "\n")
                if a.agree_log_every and astep % a.agree_log_every == 0:
                    log(f"    agreement step {astep:5d}: A_train {a_train:.4f} "
                        f"A_probe {('%.4f' % a_probe) if a_probe is not None else 'n/a'} "
                        f"ema {('%.4f' % agate.ema) if agate.ema is not None else 'n/a'} "
                        f"{'REFUSED' if agree_refuse else 'accepted'} "
                        f"(refused {astats['refused']}/{astats['checked']} so far)")
            if agree_refuse:
                # THE STEP IS SKIPPED: the batch disagreed with itself more than the model's own held-out
                # probe does, so the update is not applied. Nothing else in this step changes.
                pass
            elif gate is None:
                if opt is None:
                    # the rule this file has always used: plain SGD with a hand-set rate
                    for p in net.parameters():
                        if p.grad is not None:
                            p.data = p.data - lr_now * p.grad
                else:
                    # OUR rule: one normalised step per tensor (norm exactly opt_lr), with the decay
                    # schedule the numeric path measured as necessary (lens/runs/NUMERIC-PATH-OPT-DECAY.txt)
                    opt.step()
            else:
                # ---- THE MEMORISATION RULE, APPLIED TO THE CANDIDATE STEP (PLAN item d2b) -------------
                # The step is computed and APPLIED, then the model's own probe is re-scored; a step that
                # worsens the smoothed probe loss is REVERTED to the parameters from before it. Nothing here
                # is decided from the gradient: the model decides from its own held-out loss, which is what
                # makes this a rule it enforces on itself instead of a threshold a human picked.
                # STATED, NOT HIDDEN: with --opt relgrad a REFUSED step still advances the rule's internal
                # step counter, so its decay schedule is time-based rather than accepted-step-based. The
                # weights are restored exactly either way (np.array(B.to_cpu(...)) is a copy, not a view).
                snap = [np.array(B.to_cpu(p.data)) for p in net.parameters()]
                if opt is None:
                    for p in net.parameters():
                        if p.grad is not None:
                            p.data = p.data - lr_now * p.grad
                else:
                    opt.step()
                train_after = target_ce(net, Xtr[idx], Ytr[idx],
                                        (Wtr[idx] if Wtr is not None else None))
                probe_after = target_ce(net, Xpr, Ypr, Wpr)
                refuse, cand_t, cand_h, gap_ema = gate.evaluate(train_after, probe_after)
                gate_stats["checked"] += 1
                gstep += 1
                if refuse:
                    gate_stats["refused"] += 1
                    for p, s in zip(net.parameters(), snap):
                        p.data = B.to_dev(s)
                else:
                    gate_stats["accepted"] += 1
                    gate.commit(cand_t, cand_h)
                if gate_stats["probe_first"] is None:
                    gate_stats["probe_first"] = float(probe_after)
                gate_stats["probe_last"] = float(probe_after)
                with _open_log(gate_path) as gh:
                    gh.write(json.dumps({"step": gstep, "epoch": ep, "train_loss": float(train_after),
                                         "probe_loss": float(probe_after), "ema_train": float(cand_t),
                                         "ema_held": float(cand_h), "gap": float(gap_ema),
                                         "refused": bool(refuse), "accepted": bool(not refuse)}) + "\n")
                if a.gate_log_every and gstep % a.gate_log_every == 0:
                    log(f"    gate step {gstep:5d}: train {train_after:.4f} probe {probe_after:.4f} "
                        f"ema_train {cand_t:.4f} ema_held {cand_h:.4f} gap {gap_ema:+.4f} "
                        f"{'REFUSED' if refuse else 'accepted'} "
                        f"(refused {gate_stats['refused']}/{gate_stats['checked']} so far)")
            tot += float(B.to_cpu(parts.get("ce", total.data)))
            ctot += float(parts.get("concept_aux", 0.0))
            nb += 1
            _, lg_for_acc = net.forward(Xtr[idx])
            corr += int((B.to_cpu(lg_for_acc.data).argmax(axis=1) == Ytr[idx]).sum())
            seen += len(idx)
            if a.ckpt_seconds > 0 and (time.perf_counter() - last_ckpt) >= a.ckpt_seconds:
                _step_done = int(s_off) // int(a.batch) + 1
                last_ckpt = time.perf_counter()   # set BEFORE the attempt: a failed save waits a whole
                                                  # interval before retrying instead of retrying every step
                try:
                    _write_mid(os.path.join(OUT, f"{a.out}_mid.pkl"), net=net, opt=opt, gate=gate,
                               agate=agate, ep=ep, step_next=_step_done,
                               acc={"tot": tot, "ctot": ctot, "nb": nb, "corr": corr, "seen": seen},
                               gstep=gstep, astep=astep, gstats=gate_stats, astats=astats,
                               best_val=best_val, best_ep=best_ep, cfg_hash=cfg_hash,
                               train_windows=len(Xtr))
                    log(f"    MID-EPOCH CHECKPOINT ep {ep} step {_step_done:,}/{steps_this_ep:,} -> "
                        f"{a.out}_mid.pkl (weights + step rule + both gates + epoch totals + best memory; "
                        f"resume with --resume-mid auto)")
                except Exception as exc:   # a failed save must never kill a run, and must never be silent
                    try:
                        log(f"    MID-EPOCH CHECKPOINT FAILED ep {ep} step {_step_done:,}: "
                            f"{type(exc).__name__}: {exc} - the run continues; this failure is in the log")
                    except Exception:
                        log("    MID-EPOCH CHECKPOINT FAILED and the failure message could not itself be "
                            "formatted; see the traceback above. The run continues.")
        if a.self_context_frac > 0.0:
            log(f"  SELF-CONTEXT ep {ep}: {sctx_steps:,} self-context step(s) in total so far "
                f"({sctx_steps / max(steps_this_ep, 1) * 100:.1f}% of this epoch's {steps_this_ep:,} at the "
                f"draw p={a.self_context_frac:.3f}); round-1 own-token agreement with the TRUE next token "
                f"{sctx_hit:,}/{sctx_n:,} = {(sctx_hit / max(sctx_n, 1)) * 100:.2f}% on TRAIN windows "
                f"(the model's own error rate at step 1 of free running; the train accuracy printed below is "
                f"the TEACHER-FORCED column, unchanged, so it stays comparable to runs without this flag)")
        vl, vc, vs = 0.0, 0, 0
        if opt is not None:
            # report the step size ACTUALLY used at the last batch, not the nominal one: with the
            # per-step decay the two differ, and the printed number must be the one that ran.
            lr_now = float(opt.lr_at(max(opt.steps - 1, 0)))
        vlm, vcm, vsm = 0.0, 0, 0   # LEGACY by-token-id subset (kept; see MASKFIX note below)
        vlm_p, vcm_p, vsm_p = 0.0, 0, 0   # by-POSITION: the real assistant turns
        # MASKFIX 2026-10-22 18:0x (BUGS.md [BUG-MASK-TOKEN-ID]): `is_asst[Yva]` indexed a POSITIONAL
        # array with TOKEN IDS, so the "MASKED" column was a fixed vocabulary subset (44,418 of 87,035
        # windows, 33.9% overlap with the real assistant turns). Both are computed now; the honest one is
        # by position and it is the one checkpoint selection uses. Measured on the stored best checkpoint:
        # legacy 35.82% against by-position 17.95%, lens/runs/MASK-INDEX-BUG.txt.
        Wva = is_asst[Yva] if a.assistant_only else None
        Wva_p = is_asst[np.asarray(Sva) + a.window] if a.assistant_only else None
        for s in range(0, len(Xva), a.batch):
            idx = np.arange(s, min(s + a.batch, len(Xva)))
            va = net.forward_arrays(Xva[idx])["logits"]
            vl += float(B.to_cpu(T.softmax_ce(T.Tensor(B.to_dev(va)), Yva[idx]).data)) * len(idx)
            vc += int((va.argmax(axis=1) == Yva[idx]).sum())
            vs += len(idx)
            # Per-target cross-entropy, so the assistant-masked subset can be scored exactly the way
            # run_eval_clean.sh scores it. Before this, the trainer printed only the FULL-pool numbers,
            # which rise on every run even when the clean-val accuracy is climbing -- which is precisely
            # how a misleading self-reported number gets quoted. Now the epoch line carries both.
            if Wva is not None:
                lg = np.asarray(va, dtype=np.float64)
                mx = lg.max(axis=1, keepdims=True)
                logp = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
                ce_item = -logp[np.arange(len(idx)), Yva[idx]]
                pred_b = va.argmax(axis=1)
                w = Wva[idx] > 0
                if w.any():
                    vlm += float(ce_item[w].sum())
                    vcm += int((pred_b[w] == Yva[idx][w]).sum())
                    vsm += int(w.sum())
                wp = Wva_p[idx]
                if wp.any():
                    vlm_p += float(ce_item[wp].sum())
                    vcm_p += int((pred_b[wp] == Yva[idx][wp]).sum())
                    vsm_p += int(wp.sum())
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "concept_loss": ctot / max(nb, 1),
               "train_acc": corr / max(seen, 1), "val_ppl": float(np.exp(vl / max(vs, 1))),
               "val_acc": vc / max(vs, 1), "lr": float(lr_now),
               "opt": a.opt, "opt_lr": (opt_lr if opt is not None else None),
               "opt_momentum": a.opt_momentum, "opt_decay_final": a.opt_decay_final,
               "opt_decay_step": (opt_decay_step if opt is not None else None),
               "opt_steps": (int(opt.steps) if opt is not None else None),
               "val_masked_acc": (vcm / vsm) if vsm else None,
               "val_masked_ppl": float(np.exp(vlm / vsm)) if vsm else None,
               "val_masked_n": vsm,
               "val_masked_note": "LEGACY by-token-id subset (BUG-MASK-TOKEN-ID, pre-2026-10-22 18:0x meaning)",
               "val_asst_acc": (vcm_p / vsm_p) if vsm_p else None,
               "val_asst_ppl": float(np.exp(vlm_p / vsm_p)) if vsm_p else None,
               "val_asst_n": vsm_p,
               "val_asst_note": "BY POSITION: the target token is really inside an assistant turn (the fix)",
               "seconds": time.perf_counter() - t0}
        hist.append(rec)
        masked_txt = (f"   MASKED(legacy by-id) ppl {rec['val_masked_ppl']:.2f} "
                      f"acc {rec['val_masked_acc']*100:.2f}% (n={vsm})" if vsm else "")
        asst_txt = (f"   ASST(by-position) ppl {rec['val_asst_ppl']:.2f} "
                    f"acc {rec['val_asst_acc']*100:.2f}% (n={vsm_p})" if vsm_p else "")
        log(f"  ep {ep}/{a.epochs}  loss {rec['train_loss']:.4f} concept {rec['concept_loss']:.4f} "
            f"acc {rec['train_acc']*100:.2f}%   VAL ppl {rec['val_ppl']:.2f} "
            f"acc {rec['val_acc']*100:.2f}%{masked_txt}{asst_txt}   lr {lr_now:.4f}  ({rec['seconds']:.0f}s)")
        if gate is not None:
            log(f"    gate: checked {gate_stats['checked']:,} refused {gate_stats['refused']:,} "
                f"({gate_stats['refused'] / max(gate_stats['checked'], 1) * 100:.1f}%) accepted "
                f"{gate_stats['accepted']:,}   probe {gate_stats['probe_first']:.4f} -> "
                f"{gate_stats['probe_last']:.4f}   ema_train {gate.ema_train:.4f} ema_held {gate.ema_held:.4f}")
        if agate is not None:
            log(f"    agreement: checked {astats['checked']:,} refused {astats['refused']:,} "
                f"({astats['refused'] / max(astats['checked'], 1) * 100:.1f}%) accepted "
                f"{astats['accepted']:,}   A_train {agate.a_first:.4f} -> {agate.a_last:.4f}   "
                f"ema_A {('%.4f' % agate.ema) if agate.ema is not None else 'n/a'}   "
                f"variant {agate.variant!r} tol {agate.tol:g}")
        # MEMORISATION GATE (user's rule: the model must LEARN, never MEMORISE). A widening gap between
        # train and held-out accuracy is the signature of memorising -- measured on 2026-10-18 at 192
        # epochs: train 43.19% against honest 4.62%, a 38-point gap that looked like progress and was not.
        # With --max-gap the run STOPS there, so training cannot quietly buy memorisation, and the stop is
        # recorded in the log with both numbers rather than passing silently.
        if a.max_gap > 0 and (rec.get("val_asst_acc") is not None or rec.get("val_masked_acc") is not None):
            # MASKFIX: the gate reads the BY-POSITION column when it exists (the honest one) and says so.
            gate_col = "val_asst_acc" if rec.get("val_asst_acc") is not None else "val_masked_acc"
            gap = rec["train_acc"] - rec[gate_col]
            log(f"  memorisation check ({gate_col}): train {rec['train_acc']*100:.2f}% - held-out "
                f"{rec[gate_col]*100:.2f}% = gap {gap*100:.2f} points (limit {a.max_gap*100:.1f})")
            if gap > a.max_gap:
                log(f"  MEMORISATION GATE TRIPPED at epoch {ep}: gap {gap*100:.2f} points exceeds the "
                    f"limit {a.max_gap*100:.1f}. Stopping, because the model is storing examples instead "
                    f"of learning the rule. Checkpoint for this epoch is kept.")
                lens.save(net._net, os.path.join(OUT, f"{a.out}_ep{ep}.npz"))
                break
        lens.save(net._net, os.path.join(OUT, f"{a.out}_ep{ep}.npz"))
        # BEST-VAL CHECKPOINT (2026-10-21). Every epoch was saved, so the file anyone would ship was always
        # the LAST one - and the last is the most overfit. Measured on the tied run: it peaked at epoch 4
        # with 29.71% held-out and ended at 27.37% with perplexity tripling from 86.7 to 300.8. Shipping
        # the last epoch ships the worst of the good models. Now the peak is saved on its own, with its
        # epoch in the log, so the shipped file is the best measured one and not just the newest.
        vma = rec.get("val_asst_acc")
        col = "val_asst_acc (BY POSITION)"
        if vma is None:
            vma, col = rec.get("val_masked_acc"), "val_masked_acc (LEGACY by token id - the old, faulty column)"
        if vma is not None and (best_val is None or vma > best_val):
            best_val, best_ep = vma, ep
            lens.save(net._net, os.path.join(OUT, f"{a.out}_best.npz"))
            log(f"  NEW BEST held-out {vma*100:.2f}% at epoch {ep} [{col}] -> saved {a.out}_best.npz")
        # RNG sidecar: without it a future resume cannot reproduce the shuffle order. Cheap to
        # write (a few hundred bytes) and it is what makes resume exact rather than approximate.
        try:
            json.dump(rng.bit_generator.state,
                      open(os.path.join(OUT, f"{a.out}_ep{ep}_rng.json"), "w"))
        except (TypeError, ValueError):
            pass
        # LOGGING (fixed 2026-10-16): this used to overwrite `_log.json`, so a RESUME destroyed the
        # pre-resume history -- M0's epochs 1-29 were lost exactly that way. Now the log is MERGED by
        # epoch with whatever is already on disk, and every epoch is additionally appended to an
        # append-only JSONL that is never rewritten. A run can no longer erase its own past.
        log_path = os.path.join(OUT, f"{a.out}_log.json")
        try:
            by_ep = {r["epoch"]: r for r in json.load(open(log_path)).get("history", [])}
        except (OSError, ValueError, KeyError):
            by_ep = {}
        for r in hist:
            by_ep[r["epoch"]] = r
        json.dump({"args": {**vars(a), "data_dir": os.environ.get("CHAT_DATA_DIR", "(default)")},
                   "history": [by_ep[k] for k in sorted(by_ep)], "params": net.n_params(),
                   "concepts": spec}, open(log_path, "w"), indent=1)
        with _open_log(os.path.join(OUT, f"{a.out}_history.jsonl")) as fh:
            fh.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- THE GATE'S OWN RECORD
    if gate is not None:
        summ = {"gates": a.gates, "variant": a.gate_variant, "beta": a.gate_beta, "tol": a.gate_tol,
                "lag": a.gate_lag, "probe_windows": int(len(Xpr)), "probe_seed": int(pseed),
                "probe_heldout_pool": int(len(Xva)), "probe_disjoint_overlap": gate_stats["probe_disjoint_overlap"],
                "steps_checked": gate_stats["checked"], "steps_refused": gate_stats["refused"],
                "steps_accepted": gate_stats["accepted"],
                "refused_fraction": gate_stats["refused"] / max(gate_stats["checked"], 1),
                "probe_loss_first": gate_stats["probe_first"], "probe_loss_last": gate_stats["probe_last"],
                "ema_train_end": gate.ema_train, "ema_held_end": gate.ema_held,
                "gap_first": gate.gap_first, "gap_last": gate.gap_last,
                "per_step_log": gate_path, "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
                "opt": a.opt, "seed": a.seed}
        json.dump(summ, open(os.path.join(OUT, f"{a.out}_gates_summary.json"), "w"), indent=1)
        log("")
        log(f"  GATE RECORD: {summ['steps_refused']:,} of {summ['steps_checked']:,} steps refused "
            f"({summ['refused_fraction']*100:.1f}%), probe loss {summ['probe_loss_first']:.4f} -> "
            f"{summ['probe_loss_last']:.4f}, probe disjoint from train: "
            f"{summ['probe_disjoint_overlap']} overlapping windows")
        log(f"  written to {OUT}/{a.out}_gates_summary.json and {gate_path}")

    # ---------------------------------------------------------------- THE AGREEMENT GATE'S OWN RECORD
    if agate is not None:
        asumm = {"gates": a.gates, "variant": agate.variant, "tol": agate.tol, "beta": agate.beta,
                 "groups": agate.groups, "probe_windows": int(min(a.agree_probe_windows, len(Xpr))),
                 "probe_seed": int(pseed), "probe_heldout_pool": int(len(Xva)),
                 "probe_disjoint_overlap": gate_stats["probe_disjoint_overlap"],
                 "steps_checked": astats["checked"], "steps_refused": astats["refused"],
                 "steps_accepted": astats["accepted"],
                 "refused_fraction": astats["refused"] / max(astats["checked"], 1),
                 "a_train_first": agate.a_first, "a_train_last": agate.a_last, "ema_a_end": agate.ema,
                 "independence_floor": 1.0 / np.sqrt(max(agate.groups, 1)),
                 "per_step_log": agate_path, "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
                 "opt": a.opt, "seed": a.seed}
        json.dump(asumm, open(os.path.join(OUT, f"{a.out}_agreement_summary.json"), "w"), indent=1)
        log("")
        log(f"  AGREEMENT RECORD: {asumm['steps_refused']:,} of {asumm['steps_checked']:,} steps refused "
            f"({asumm['refused_fraction']*100:.1f}%), A_train {asumm['a_train_first']:.4f} -> "
            f"{asumm['a_train_last']:.4f}, variant {asumm['variant']!r} tol {asumm['tol']:g}")
        log(f"  written to {OUT}/{a.out}_agreement_summary.json and {agate_path}")

    # ---------------------------------------------------------------- THE REAL TEST
    log("")
    log("=" * 96)
    log("CAN IT TALK? — what it says, and what the audit says about its concepts")
    log("=" * 96)
    vmap = {w: i for i, w in enumerate(vocab)}

    def encode(t):
        toks = re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", t.lower())
        return [vmap.get(x, 0) for x in toks]

    def generate(prompt, n_new=40, temp=0.8, greedy=False, seed=0, top_k=40, rep_penalty=1.15):
        """Generate a reply.

        TOP-K AND REPETITION PENALTY ARE NEW, and they matter for output quality independently of
        the model. Greedy decoding collapses to a single repeated phrase ("i'm not sure i'm not
        sure") because the highest-probability continuation is a loop; pure sampling at temp 0.8
        drifts into word salad. Top-k keeps sampling inside the plausible set, and the repetition
        penalty subtracts from tokens already used in this reply so the model cannot loop.

        These change the DECODER, not the model. The model's perplexity and accuracy are unaffected.
        """
        enc = encode(prompt)
        # PAD-FREE CONTEXT (fixed 2026-11-06; measured in lens/runs/PROMPT-ENCODING-AUDIT.txt).
        # This used to be `enc = [0]*(window-len(enc)) + enc`, i.e. id 0 = "<unk>". The six prompts
        # below are only 9-12 ids long, so 20-23 of the 32 context slots were <unk>, and the model
        # then emitted <unk> for 95.1-96.5% of its reply tokens. Repeating the prompt to fill the
        # window (same weights, same decoder) took that to 26.4/21.5%. Predictions and numbers:
        # docs/history/PROMPT-ENCODING.md. Do not reintroduce id-0 padding here.
        if a.window > 1 and enc:
            enc = (enc * ((a.window // len(enc)) + 2))[-a.window:]
        ctx = enc[-a.window:]
        out, r = [], np.random.default_rng(seed)
        used = {}
        for _ in range(n_new):
            _, lg = net.forward(np.asarray([ctx], dtype=np.int64))
            z = B.to_cpu(lg.data)[0].astype(np.float64)
            for tok, cnt in used.items():
                z[tok] -= np.log(rep_penalty) * cnt
            if greedy:
                nxt = int(z.argmax())
            else:
                zz = z / max(temp, 1e-6)
                # top-k: keep only the k highest-scoring tokens, renormalise
                if top_k and top_k < len(zz):
                    cut = np.partition(zz, -top_k)[-top_k]
                    zz = np.where(zz >= cut, zz, -np.inf)
                zz = zz - np.nanmax(zz[np.isfinite(zz)])
                p = np.exp(zz - zz.max())
                p = np.where(np.isfinite(p), p, 0.0)
                s = p.sum()
                p = p / s if s > 0 else np.ones(len(p)) / len(p)
                nxt = int(r.choice(len(p), p=p))
            out.append(vocab[nxt] if nxt < len(vocab) else "<unk>")
            used[nxt] = used.get(nxt, 0) + 1
            ctx = (ctx + [nxt])[-a.window:]
        return " ".join(out)

    prompts = [
        "Human: hello , how are you ?\n\nAssistant:",
        "Human: what is your name ?\n\nAssistant:",
        "Human: can you help me learn to cook ?\n\nAssistant:",
        "Human: i am feeling sad today .\n\nAssistant:",
        "Human: what is the capital of france ?\n\nAssistant:",
        "Human: tell me about dogs .\n\nAssistant:",
    ]
    replies = []
    for p in prompts:
        g = generate(p, n_new=45, greedy=True)
        s = generate(p, n_new=45, temp=0.8, seed=1, top_k=40, rep_penalty=1.15)
        s2 = generate(p, n_new=45, temp=0.7, seed=2, top_k=20, rep_penalty=1.3)
        log(f"\n  PROMPT   {p.splitlines()[0]!r}")
        log(f"  GREEDY   {g}")
        log(f"  TOPK40   {s}")
        log(f"  TOPK20   {s2}")
        replies.append({"prompt": p, "greedy": g, "sampled": s, "sampled_topk20": s2})

    log("")
    log("  --- concept fidelity audit (does each name mean what it says?) ---")
    aud = lens.audit(net._net, Xva[:4000], concepts=concepts)
    for c in aud["concepts"]:
        au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
        fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
        log(f"     {c['name']:<22} AUROC {au:>7}  fires {fr:>6}  {'PASS' if c['pass'] else 'FAIL'}")
    log(f"     -> {aud['n_pass']}/{aud['n_concepts']} pass   coverage {aud['coverage']*100:.2f}%")

    json.dump({"replies": replies, "audit": aud, "history": hist,
               "params": net.n_params(), "concepts": spec},
              open(os.path.join(OUT, f"{a.out}_replies.json"), "w"), indent=1)
    log("")
    log("=" * 96)
    log(f"  final VAL ppl {hist[-1]['val_ppl']:.2f}  acc {hist[-1]['val_acc']*100:.2f}%   "
        f"concepts {aud['n_pass']}/{aud['n_concepts']}")
    log(f"  saved to {OUT}/{a.out}_*")
    log("=" * 96)


if __name__ == "__main__":
    main()
