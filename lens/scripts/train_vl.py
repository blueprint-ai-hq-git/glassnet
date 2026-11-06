"""train_vl.py — the VISION-LANGUAGE model: image + question -> answer.

THE ARCHITECTURE
----------------
    image (28x28) -> 16 patches x D  --\
                                        >-- ONE SHARED concept layer -> answer layer -> answer word
    question       ->  L tokens   x D --/

The concept layer holds BOTH kinds of concept:
    image concepts  is_bright, left_right_symmetric, has_fine_texture, ...   (from PIXELS)
    text concepts   asks_what, asks_name, asks_see, asks_identify            (from the QUESTION)
and the answer layer reads all of them. So the model can only answer through named, checkable
properties on both sides -- which is what makes this a readable VL model and not two models glued.

WHY THE QUESTIONS VARY
----------------------
With a single question template the text concepts would be constant and carry no information. Four
different ways of asking for the same thing means the text half genuinely varies, and the answer
layer has to combine "which question is this" with "what is in the picture".

THE METRIC IS CHECKABLE
-----------------------
The generated answer either names the right class or it does not. No interpretation, no judge.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import image_concepts                                    # noqa: E402
import lens.tensor as T                                  # noqa: E402
import vl_data                                           # noqa: E402
from lens import backend as B                            # noqa: E402
from lens.model import Dense                             # noqa: E402
from lens.vision import TextConcept, VLNet               # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "vl")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


def audit_vl(net, X, Q, y, meta):
    """Does each concept -- image side AND text side -- mean what its name says?

    AUROC is computed per concept against its OWN formula, on the same samples the model sees.
    """
    out = net.forward_arrays(X, Q)
    h = out["concepts"]
    Aw = B.to_cpu(net.answer.W.data)
    targets = net.concept_targets(X, Q)
    res = []
    for k, c in enumerate(net.concepts):
        lab = targets[:, k]
        a = h[:, k]
        pos, neg = a[lab > 0.5], a[lab < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            au = None
        else:
            order = np.argsort(np.concatenate([pos, neg]))
            ranks = np.empty(len(order), dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            rp = ranks[:len(pos)].sum()
            au = float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        fires = float((a > 0.5).mean())
        contrib = float(np.abs(Aw[k]).mean() * a.mean())
        passed = (au is not None and au >= 0.55) and (fires >= 0.02) and (contrib >= 0.05)
        side = "image" if k < len(net.img_concepts) else "text "
        res.append({"name": c.name, "side": side, "auroc": au, "fires": fires,
                    "contrib": contrib, "support": float(lab.mean()), "pass": bool(passed)})
    n_img = sum(r["pass"] for r in res if r["side"] == "image")
    n_txt = sum(r["pass"] for r in res if r["side"] == "text ")
    return {"concepts": res, "n_pass": sum(r["pass"] for r in res), "n_concepts": len(res),
            "img_pass": n_img, "img_total": len(net.img_concepts),
            "txt_pass": n_txt, "txt_total": len(net.txt_concepts)}


class PlainVQA:
    """Control: same job, no named concepts. Image + question concatenated into one hidden layer."""

    def __init__(self, n_in, hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.l1 = Dense(n_in, hidden, rng, name="p1")
        self.l2 = Dense(hidden, n_out, rng, name="p2")

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def forward(self, X):
        x = T.Tensor(B.to_dev(np.asarray(X, dtype=np.float32)))
        return self.l2(T.relu(self.l1(x)))

    def loss(self, X, y, class_weights=None):
        return T.softmax_ce(self.forward(X), y, class_weights=class_weights), {"ce": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="vqa1", choices=["vqa1", "vqa2", "vqa3"],
                    help="vqa1 = questions all ask the same thing; vqa2/vqa3 = questions matter "
                         "(vqa3 picks dataset-appropriate questions and validates the predicates)")
    ap.add_argument("--data", default="fashion", help="fashion | pathmnist | ...")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--color", action="store_true",
                    help="keep the RGB channels for RGB datasets instead of averaging them. "
                         "PathMNIST channel means are 188.8/136.0/180.0 and stain colour separates "
                         "tissue types, so averaging destroys the discriminating signal")
    ap.add_argument("--colour-concepts", dest="colour_concepts", action="store_true",
                    help="add concepts measured on the RGB CHANNELS (is_pink_dominant, "
                         "is_purple_dominant, green_channel_low...). Requires --color. Without this "
                         "the model consumes colour but the readable layer cannot NAME it")
    ap.add_argument("--adaptive", action="store_true",
                    help="fit concept thresholds on TRAIN as quantiles, instead of the fixed "
                         "Fashion-MNIST thresholds that made 3 of 12 concepts near-constant on "
                         "PathMNIST and drove pos_weight up to 20")
    ap.add_argument("--balance", action="store_true",
                    help="weight the answer loss by inverse label frequency, so a model cannot "
                         "score the majority baseline by always saying the commonest answer")
    ap.add_argument("--out", default="vl1")
    ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()

    log("=" * 100)
    log("THE VISION-LANGUAGE MODEL — image + question -> answer, through ONE shared readable layer")
    log("=" * 100)
    T.set_device(a.device)

    import vl_data
    if a.task == "vqa3":
        import vl_data3
        which_set = "medical" if (a.data != "fashion" and os.environ.get("VLQ", "auto") == "auto") \
            else os.environ.get("VLQ", "fashion")
        base = vl_data.prepare_vqa(a.data, max_train=None, seed=a.seed, color=a.color)
        if base is None:
            log(f"  data '{a.data}' unavailable")
            return
        d = vl_data3.prepare(base, which_set, seed=a.seed)
        if a.max_train:
            sel = np.random.default_rng(a.seed).permutation(len(d["Xtr"]))[:a.max_train]
            for k in ("Xtr", "ytr", "Qtr", "Atr"):
                d[k] = d[k][sel]
            d["meta"]["n_train"] = len(d["Xtr"])
    elif a.task == "vqa2":
        import vl_data2
        base = vl_data.prepare_vqa(a.data, max_train=None, seed=a.seed, color=a.color)
        d = vl_data2.prepare_vqa2(base, max_train=(a.max_train or None), seed=a.seed)
    else:
        d = vl_data.prepare_vqa(a.data, max_train=(a.max_train or None), seed=a.seed, color=a.color)
    if d is None:
        log(f"  data '{a.data}' unavailable")
        return
    m = d["meta"]
    Xtr, Qtr, Atr, ytr = d["Xtr"], d["Qtr"], d["Atr"], d["ytr"]
    Xte, Qte, Ate, yte = d["Xte"], d["Qte"], d["Ate"], d["yte"]
    qlen, avocab = m["qlen"], d["avocab"]
    log(f"  source {m['source']}")
    log(f"  train {Xtr.shape}  test {Xte.shape}   questions {m['questions']}")
    log(f"  answer vocab ({len(avocab)}): {avocab}")
    log(f"  image is {Xtr.shape[1]} pixels in [{Xtr.min():.2f},{Xtr.max():.2f}]")

    # ---------------------------------------------------------------- the two halves of the layer
    if a.adaptive:
        import image_concepts_adaptive as ic_mod
    else:
        import image_concepts as ic_mod
    img_concepts, cspec = ic_mod.build(Xtr, min_support=0.02, verbose=False)
    if os.environ.get("COLOUR_ONLY"):
        # MED-2 (2026-10-13): the strict audit measured the grayscale concepts 12/13
        # mislabelled/weak on colour data — the neurons latch the RGB axes and the
        # channel-averaged formulas cannot survive the rivalry. Histology's signal IS
        # colour; build the readable layer from the colour concepts alone.
        img_concepts = []
        cspec = {"kept": 0, "dropped": len(img_concepts), "n_defined": 0}
    log(f"  image concepts: {cspec['kept']} kept, {cspec['dropped']} dropped of {cspec['n_defined']}")
    if cspec.get("adaptive"):
        for r in cspec["spec"][:4]:
            log(f"    {r['name']:<24} support {r['support']:.4f}  pw {r['pos_weight']:.2f}  "
                f"{r['formula']}")
        for r in cspec.get("dropped_detail", [])[:4]:
            log(f"    DROPPED {r['name']:<20} {r['reason']}")

    # --- COLOUR concepts: measured on the RGB channels, so the readable layer can NAME the signal
    #     rather than only consuming it. Requires the data to have kept colour.
    if a.colour_concepts:
        import colour_concepts
        ckept, cspec_c = colour_concepts.build(Xtr, min_support=0.02, verbose=False)
        if os.environ.get("COLOUR_SELECT"):
            # MED-3 (2026-10-13): measured collinearity wall — 12/105 colour pairs > 0.8
            # formula-corr on H&E (one pair = 1.000 identical), so the 15-set cannot pass
            # the strict 0.10 margin by construction. COLOUR_SELECT = the pruned set
            # (max-pair |corr| <= 0.565), chosen from the audit datum.
            sel = [n.strip() for n in os.environ["COLOUR_SELECT"].split(",")]
            ckept = [c for c in ckept if c.name in sel]
            cspec_c = {**cspec_c, "kept": len(ckept),
                       "spec": [s for s in cspec_c.get("spec", []) if s["name"] in sel]}
        log("")
        log(f"  colour concepts: {cspec_c['kept']} kept, {cspec_c['dropped']} dropped")
        for r in cspec_c.get("spec", []):
            log(f"    {r['name']:<26} support {r['support']:.4f}  pw {r['pos_weight']:.2f}  "
                f"{r['formula']}")
        for r in cspec_c.get("dropped_detail", [])[:4]:
            log(f"    DROPPED {r['name']:<24} {r.get('reason', '')}")
        img_concepts = list(img_concepts) + list(ckept)

    # text concepts: the DISTINGUISHING word of each question template, so each one varies
    txt_concepts = []
    for q, cname in zip(m["questions"], m["qconcepts"]):
        toks = cname.replace("asks_", "").split("_")
        ids = [d["qvmap"][w] for w in toks if w in d["qvmap"]]
        if not ids:
            continue
        supp = float(np.isin(Qtr, ids).any(axis=1).mean())
        if supp < 0.02:
            log(f"    DROPPED text concept {cname} support {supp:.4f}")
            continue
        txt_concepts.append(TextConcept(cname, ids, support=supp,
                                        formula=f"question contains {toks}"))
    log(f"  text concepts: {len(txt_concepts)}")
    for c in txt_concepts:
        log(f"    {c.name:<18} support {c.support:.4f}   {c.formula}")
    log(f"  >>> ONE layer with {len(img_concepts)} image concepts + {len(txt_concepts)} text concepts")

    ch = 3 if (d["Xtr"].shape[1] == 2352) else 1
    net = VLNet(img_concepts, txt_concepts, q_vocab_size=len(d["qvocab"]),
                answer_vocab_size=len(avocab), caption_len=qlen, channels=ch,
                d=a.d, patch=a.patch, seed=a.seed, lam_concept=a.lam_concept)
    npar = int(sum(p.data.size for p in net.parameters()))
    log("")
    log(f"  READABLE: {npar:,} params   {net.P} patches ({ch} channel"
        f"{'s' if ch > 1 else ''}, {net.front.patch_dim} values each) + {qlen} question tokens -> "
        f"{net.K} named concepts -> {len(avocab)} answers")

    # matched plain control: image + question one-hot -> hidden -> answer
    n_in = int(d["Xtr"].shape[1]) + len(d["qvocab"])
    hidden = max(2, round((npar - len(avocab) - 10) / (n_in + len(avocab))))
    plain = PlainVQA(n_in, hidden, len(avocab), seed=a.seed)
    ppar = int(sum(p.data.size for p in plain.parameters()))
    log(f"  PLAIN   : {ppar:,} params  784+onehot -> {hidden} -> {len(avocab)}   "
        f"({abs(ppar-npar)/npar*100:.2f}% size difference)")

    def onehot_q(Q):
        z = np.zeros((len(Q), len(d["qvocab"])), dtype=np.float32)
        z[np.arange(len(Q)), Q[:, 0]] = 1.0
        return z

    # --- class weights for the answer loss, computed on TRAIN answers only
    cw = None
    if a.balance:
        cnt = np.bincount(Atr[:, 0], minlength=len(avocab)).astype(np.float64)
        cnt = np.maximum(cnt, 1.0)
        cw = (cnt.sum() / (len(avocab) * cnt)).astype(np.float32)
        log("")
        log("  BALANCED ANSWER LOSS — inverse-frequency weights (train answers only):")
        order = np.argsort(-cw)
        for i in order[:6]:
            log(f"    {avocab[int(i)]:<26} count {int(cnt[int(i)]):>6}  weight {cw[int(i)]:.3f}")
        log(f"    (the majority answer would otherwise be worth "
            f"{cnt.max()/cnt.sum()*100:.2f}% for free)")

    def plain_X(X, Q):
        return np.concatenate([X, onehot_q(Q)], axis=1).astype(np.float32)

    results = {}
    for tag in ("readable", "plain"):
        log("")
        log(f"  --- training {tag} ---")
        rng = np.random.default_rng(a.seed)
        hist = []
        for ep in range(1, a.epochs + 1):
            t0 = time.perf_counter()
            order = rng.permutation(len(Xtr))
            lr_now = a.lr * (1.0 - a.lr_decay * (ep - 1) / max(a.epochs - 1, 1))
            tot, nb = 0.0, 0
            for s in range(0, len(Xtr), a.batch):
                idx = order[s:s + a.batch]
                params = net.parameters() if tag == "readable" else plain.parameters()
                for p in params:
                    p.grad = None
                if tag == "readable":
                    loss, parts = net.loss(Xtr[idx], Qtr[idx], Atr[idx][:, 0], class_weights=cw)
                else:
                    loss, parts = plain.loss(plain_X(Xtr[idx], Qtr[idx]), Atr[idx][:, 0],
                                             class_weights=cw)
                loss.backward()
                for p in params:
                    if p.grad is not None:
                        p.data = p.data - lr_now * p.grad
                tot += float(B.to_cpu(loss.data))
                nb += 1

            def answer_acc(X, Q, A):
                right = 0
                for s in range(0, len(X), 512):
                    sl = np.arange(s, min(s + 512, len(X)))
                    if tag == "readable":
                        lg = net.forward_arrays(X[sl], Q[sl])["logits"]
                    else:
                        lg = B.to_cpu(plain.forward(plain_X(X[sl], Q[sl])).data)
                    right += int((lg.argmax(axis=1) == A[sl, 0]).sum())
                return right / len(X)

            tra = answer_acc(Xtr[:8000], Qtr[:8000], Atr[:8000])
            tea = answer_acc(Xte, Qte, Ate)
            rec = {"epoch": ep, "loss": tot / max(nb, 1), "train_acc": tra, "test_acc": tea,
                   "lr": lr_now, "seconds": time.perf_counter() - t0}
            hist.append(rec)
            log(f"    ep {ep:>2}/{a.epochs}  loss {rec['loss']:.4f}  train {tra*100:6.2f}%  "
                f"TEST {tea*100:6.2f}%  lr {lr_now:.3f}  ({rec['seconds']:.0f}s)")
        results[tag] = {"history": hist, "params": npar if tag == "readable" else ppar}
        if tag == "readable":
            from lens.vision import save_vision
            # `ep` leaks from the loop but naming the file from `a.epochs` is explicit and cannot
            # silently label a checkpoint with the wrong epoch if the loop body changes later.
            save_vision(net, os.path.join(OUT, f"{a.out}_ep{a.epochs}.npz"), kind="vl",
                        avocab=avocab, qvocab=d["qvocab"])

    # ---------------------------------------------------------------- the audit
    log("")
    log("=" * 100)
    log("THE FIDELITY AUDIT — every concept, both sides of the fused layer")
    log("=" * 100)
    aud = audit_vl(net, Xte, Qte, yte, m)
    for c in aud["concepts"]:
        au = "None " if c["auroc"] is None else f"{c['auroc']:.4f}"
        log(f"    [{c['side']}] {c['name']:<24} AUROC {au}  fires {c['fires']:>6.3f}  "
            f"contrib {c['contrib']:.3f}  {'PASS' if c['pass'] else 'FAIL'}")
    log(f"    -> {aud['n_pass']}/{aud['n_concepts']}  "
        f"(image {aud['img_pass']}/{aud['img_total']}, text {aud['txt_pass']}/{aud['txt_total']})")

    # --- the number this run exists to move: how faithful is the TEXT half?
    def mean_auroc(side):
        v = [c["auroc"] for c in aud["concepts"] if c["side"] == side and c["auroc"] is not None]
        return float(np.mean(v)) if v else float("nan")

    log("")
    log("  --- AUROC BY SIDE (0.5 = coin flip) ---")
    log(f"    image concepts  mean AUROC {mean_auroc('image'):.4f}")
    log(f"    text  concepts  mean AUROC {mean_auroc('text '):.4f}")
    log(f"    (vqa1 baseline to beat: text concepts sat at 0.53-0.63)")

    log("")
    log("=" * 100)
    log("READABLE vs PLAIN")
    log("=" * 100)
    r, p = results["readable"], results["plain"]
    log(f"    readable  {r['params']:>8,} params   TEST {r['history'][-1]['test_acc']*100:6.2f}%")
    log(f"    plain     {p['params']:>8,} params   TEST {p['history'][-1]['test_acc']*100:6.2f}%")
    delta = (r['history'][-1]['test_acc'] - p['history'][-1]['test_acc']) * 100
    log(f"    delta {delta:+.2f} points")

    # --- the majority-answer baseline. On vqa2 "no" is 42.5% of answers, so a model that ALWAYS
    #     says "no" scores 42.5%. Without this line, 43% would look like a working model.
    u, cnt = np.unique(Ate[:, 0], return_counts=True)
    maj = float(cnt.max() / cnt.sum())
    maj_word = avocab[int(u[int(np.argmax(cnt))])]
    log("")
    log(f"    majority-answer baseline: always say {maj_word!r} = {maj*100:.2f}%")
    log(f"    true chance (1/{len(avocab)}) = {100/len(avocab):.2f}%")
    beats_maj = r['history'][-1]['test_acc'] > maj
    log(f"    readable beats the majority baseline: {beats_maj} "
        f"({r['history'][-1]['test_acc']*100:.2f}% vs {maj*100:.2f}%)")

    # ---------------------------------------------------------------- SHOW the answers
    log("")
    log("=" * 100)
    log("WHAT IT ANSWERS — generated, verbatim")
    log("=" * 100)
    shown = 0
    for i in range(len(Xte)):
        if shown >= 12:
            break
        q = " ".join(d["qvocab"][t] for t in Qte[i] if t != 0)
        lg = net.forward_arrays(Xte[i:i+1], Qte[i:i+1])["logits"][0]
        pred = avocab[int(lg.argmax())]
        true = avocab[int(Ate[i, 0])]
        if shown < 8 or pred != true:
            log(f"    Q: {q:<24} ANSWER: {pred:<24} TRUE: {true:<24} "
                f"{'OK' if pred == true else 'wrong'}")
            shown += 1

    json.dump({"args": vars(a), "meta": m, "concept_spec": cspec, "audit": aud, "results": results,
               "readable_params": npar, "plain_params": ppar,
               "tokens": {"qvocab": d["qvocab"], "avocab": avocab}},
              open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)
    log("")
    log(f"  final: TEST {r['history'][-1]['test_acc']*100:.2f}%   "
        f"concepts {aud['n_pass']}/{aud['n_concepts']}   saved {a.out}_log.json")
    log("=" * 100)


if __name__ == "__main__":
    main()
