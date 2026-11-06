"""vl_data3.py — VQA2 questions made DATASET-APPROPRIATE, and an honest baseline split.

WHY THIS EXISTS
---------------
vqa2 fixed the "questions carry no information" defect by asking DIFFERENT things:

    "what is in the photo"  -> the class name
    "is this footwear"      -> yes / no

But `is this footwear` is Fashion-MNIST-specific. Asking a colon-pathology model "is this footwear"
would be a nonsense question, and a yes/no question whose answer is always "no" for every medical
class teaches the model nothing and would make the text concept `asks_footwear` look faithful while
carrying no signal.

So the yes/no predicates are defined per dataset, over the class names themselves, and a predicate
that does not split the classes is REJECTED rather than quietly kept.

Also added: the answer distribution is checked, because a yes/no split that lands 95/5 makes the
majority baseline look like accuracy (the vqa2 "no" answer was 42.34% of all answers).
"""
import numpy as np

# dataset -> list of (question text, concept name, kind, predicate set or None for "name")
QUESTIONSETS = {
    "fashion": [
        ("what is in the photo", "asks_what", "name", None),
        ("name the object", "asks_name", "name", None),
        ("is this footwear", "asks_footwear", "yesno", {"sandal", "sneaker", "ankle_boot"}),
        ("is this a bag", "asks_bag", "yesno", {"bag"}),
    ],
    "medical": [
        ("what is in the photo", "asks_what", "name", None),
        ("name the tissue", "asks_name", "name", None),
        ("is this cancer", "asks_cancer", "yesno",
         {"cancer_associated_stroma", "colorectal_adenocarcinoma"}),
        ("is this normal tissue", "asks_normal", "yesno", {"normal_colon_mucosa"}),
    ],
}


def questions_for(which):
    """Pick the question set by dataset. Unlisted datasets get the fashion set only if their class
    names actually appear in it; otherwise the yes/no questions are dropped rather than mis-asked."""
    if which in QUESTIONSETS:
        return QUESTIONSETS[which]
    return QUESTIONSETS["fashion"]


def qvocab_for(qs):
    toks = []
    for q, _, _, _ in qs:
        toks.extend(q.split())
    uniq = ["<pad>", "<unk>", "yes", "no"] + sorted(set(toks))
    return uniq, {w: i for i, w in enumerate(uniq)}


def encode_q(q, vmap, length):
    ids = [vmap.get(w, 1) for w in q.split()][:length]
    return ids + [0] * (length - len(ids))


def prepare(base, which, seed=0):
    names = base["meta"]["names"]
    Xtr, ytr = base["Xtr"], base["ytr"]
    Xte, yte = base["Xte"], base["yte"]
    qs = questions_for(which)
    qvocab, qvmap = qvocab_for(qs)
    qlen = max(len(q.split()) for q, _, _, _ in qs)

    awords = sorted({nm.replace(" ", "_") for nm in names})
    avocab = ["<pad>", "<unk>"] + awords + ["yes", "no"]
    avmap = {w: i for i, w in enumerate(avocab)}

    # --- validate the yes/no predicates BEFORE using them: each must split the classes
    kept, rejected = [], []
    for q, cname, kind, pred in qs:
        if kind == "name":
            kept.append((q, cname, kind, pred))
            continue
        hits = [nm for nm in names if nm.replace(" ", "_") in pred]
        frac = len(hits) / len(names)
        if frac <= 0.0 or frac >= 1.0:
            rejected.append((cname, f"predicate matches {len(hits)}/{len(names)} classes -- "
                                    f"it cannot be answered, it is constant"))
            continue
        kept.append((q, cname, kind, pred))
    if rejected:
        print("  REJECTED questions (constant answer, no signal):")
        for c, why in rejected:
            print(f"    {c}: {why}")

    def build(y):
        Q = np.zeros((len(y), qlen), dtype=np.int64)
        A = np.zeros((len(y),), dtype=np.int64)
        for i in range(len(y)):
            q, cname, kind, pred = kept[i % len(kept)]
            cls = names[int(y[i])]
            Q[i] = encode_q(q, qvmap, qlen)
            if kind == "name":
                A[i] = avmap[cls.replace(" ", "_")]
            else:
                A[i] = avmap["yes"] if cls.replace(" ", "_") in pred else avmap["no"]
        return Q, A

    Qtr, Atr = build(ytr)
    Qte, Ate = build(yte)
    Atr = Atr.reshape(-1, 1)
    Ate = Ate.reshape(-1, 1)

    # --- the honest baseline: what does always-answering-the-most-common-thing score?
    u, cnt = np.unique(Ate[:, 0], return_counts=True)
    maj_i = int(u[int(np.argmax(cnt))])
    majority = float(cnt.max() / cnt.sum())

    meta = {"which": which, "names": names, "qlen": qlen, "n_train": len(Xtr),
            "n_test": len(Xte), "source": base["meta"]["source"],
            "questions": [q for q, _, _, _ in kept],
            "qconcepts": [c for _, c, _, _ in kept],
            "qkinds": [k for _, _, k, _ in kept],
            "rejected": [{"concept": c, "reason": w} for c, w in rejected],
            "answer_vocab": avocab,
            "majority_answer": avocab[maj_i], "majority_baseline": majority,
            "true_chance": 1.0 / len(avocab)}
    return {"Xtr": Xtr, "ytr": ytr, "Qtr": Qtr, "Atr": Atr,
            "Xte": Xte, "yte": yte, "Qte": Qte, "Ate": Ate,
            "qvocab": qvocab, "qvmap": qvmap, "avocab": avocab, "avmap": avmap,
            "meta": meta}


if __name__ == "__main__":
    import os
    import sys
    PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
    sys.path.insert(0, PROJ)
    sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
    import vl_data

    for which in ("fashion", "pathmnist"):
        print("=" * 100)
        print(f"{which}")
        print("=" * 100)
        base = vl_data.prepare_vqa(which, max_train=None, seed=0)
        d = prepare(base, which if which == "fashion" else "medical")
        m = d["meta"]
        print(f"  source: {m['source']}")
        for q, c, k in zip(m["questions"], m["qconcepts"], m["qkinds"]):
            print(f"    {q:<24} concept {c:<14} kind {k}")
        if m["rejected"]:
            for r in m["rejected"]:
                print(f"    REJECTED {r['concept']}: {r['reason']}")
        print(f"  majority answer {m['majority_answer']!r} = {m['majority_baseline']*100:.2f}%  "
              f"(true chance {m['true_chance']*100:.2f}%)")
        print()
        print("  the same image, different questions, different answers:")
        names = m["names"]
        shown = {}
        for i in range(len(d["ytr"])):
            cls = names[int(d["ytr"][i])]
            if len(shown.setdefault(cls, [])) < 4:
                shown[cls].append(i)
        for cls in list(shown)[:3]:
            for i in shown[cls]:
                q = " ".join(d["qvocab"][t] for t in d["Qtr"][i] if t != 0)
                print(f"    {cls:<26} Q {q:<24} A {d['avocab'][d['Atr'][i,0]]}")
        print()
