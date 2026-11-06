"""vl_data2.py — VQA where the QUESTION MATTERS.

THE MEASURED PROBLEM THIS FIXES
-------------------------------
In the first VL design every question asked for the same thing:

    "what is in the photo" -> sneaker
    "name the object"      -> sneaker
    "what do you see"      -> sneaker
    "identify this item"   -> sneaker

The question carried NO information about the target, so the model had no reason to read it, and the
text concepts got no gradient pressure. Measured result: **every text concept sat at AUROC 0.53-0.63
-- barely above a coin flip** -- while image concepts reached 0.95-1.00. The text half of the "fused"
layer was decorative.

THE FIX, AS ONE CHANGE
----------------------
Make different questions need DIFFERENT answers:

    "what is in the photo"  -> the class name      ("sneaker")
    "name the object"       -> the class name      ("sneaker")
    "is this footwear"      -> yes / no            (sandal, sneaker, ankle boot = yes)
    "is this a bag"         -> yes / no            (bag = yes)

Now the model CANNOT answer without reading the question, so the text concepts become load-bearing.

Pre-registered prediction: text-concept AUROC rises clearly above 0.70, and above the previous
0.53-0.63 band, because the concepts now carry information the answer layer needs. If it does not
rise, the explanation is wrong and the weakness lies somewhere else.
"""
import numpy as np

QUESTIONS2 = [
    ("what is in the photo", "asks_what", "name"),
    ("name the object", "asks_name", "name"),
    ("is this footwear", "asks_footwear", "yesno"),
    ("is this a bag", "asks_bag", "yesno"),
]

# which Fashion-MNIST classes are footwear / a bag
FOOTWEAR = {"sandal", "sneaker", "ankle_boot"}
IS_BAG = {"bag"}


def question_vocab2():
    toks = []
    for q, _, _ in QUESTIONS2:
        toks.extend(q.split())
    uniq = ["<pad>", "<unk>", "yes", "no"] + sorted(set(toks))
    return uniq, {w: i for i, w in enumerate(uniq)}


def encode_question2(q, vmap, length):
    ids = [vmap.get(w, 1) for w in q.split()][:length]
    return ids + [0] * (length - len(ids))


def prepare_vqa2(base, max_train=None, seed=0):
    """Build the load-bearing-question task on top of a prepared image/label set.

    `base` is the dict from vl_data.prepare_vqa(which, max_train=...) -- we only need its images,
    labels and class names, and we rebuild the questions and answers here.
    """
    names = base["meta"]["names"]
    Xtr, ytr = base["Xtr"], base["ytr"]
    Xte, yte = base["Xte"], base["yte"]

    qvocab, qvmap = question_vocab2()
    qlen = max(len(q.split()) for q, _, _ in QUESTIONS2)

    # answers: the class name, plus yes/no
    awords = sorted({nm.replace(" ", "_") for nm in names})
    avocab = ["<pad>", "<unk>"] + awords + ["yes", "no"]
    avmap = {w: i for i, w in enumerate(avocab)}

    def answer_for(cls_name, kind):
        if kind == "name":
            return avmap[cls_name.replace(" ", "_")]
        if kind == "yesno":
            if "footwear" in " ".join(q for q, _, k in QUESTIONS2 if k == "yesno"):
                pass
            return None                                  # filled in by the caller per template
        return avmap["no"]

    # assign templates round-robin so every question appears equally often
    def build(y):
        Q = np.zeros((len(y), qlen), dtype=np.int64)
        A = np.zeros((len(y),), dtype=np.int64)
        qnames = []
        for i in range(len(y)):
            q, cname, kind = QUESTIONS2[i % len(QUESTIONS2)]
            cls = names[int(y[i])]
            Q[i] = encode_question2(q, qvmap, qlen)
            if kind == "name":
                A[i] = avmap[cls.replace(" ", "_")]
            elif cname == "asks_footwear":
                A[i] = avmap["yes"] if cls in FOOTWEAR else avmap["no"]
            else:                                        # asks_bag
                A[i] = avmap["yes"] if cls in IS_BAG else avmap["no"]
            qnames.append(cname)
        return Q, A, qnames

    Qtr, Atr, _ = build(ytr)
    Qte, Ate, _ = build(yte)
    # match vl_data.prepare_vqa's shape: (N, 1). The trainer indexes Atr[:, 0], and a 1-D array
    # raises "too many indices for array".
    Atr = Atr.reshape(-1, 1)
    Ate = Ate.reshape(-1, 1)

    meta = {"which": base["meta"]["which"], "names": names, "qlen": qlen,
            "n_train": len(Xtr), "n_test": len(Xte),
            "source": base["meta"]["source"],
            "questions": [q for q, _, _ in QUESTIONS2],
            "qconcepts": [c for _, c, _ in QUESTIONS2],
            "qkinds": [k for _, _, k in QUESTIONS2],
            "answer_vocab": avocab}
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

    base = vl_data.prepare_vqa("fashion", max_train=None, seed=0)
    d = prepare_vqa2(base, max_train=2000)
    m = d["meta"]
    print(f"  questions (all four must appear, and answer DIFFERENTLY):")
    for q, c, k in zip(m["questions"], m["qconcepts"], m["qkinds"]):
        print(f"    {q:<22} concept {c:<14} kind {k}")
    print()
    print(f"  answer vocab ({len(d['avocab'])}): {d['avocab']}")
    print()
    print("  the point: the SAME image gets different answers depending on the question")
    y = d["ytr"]
    shown = {}
    for i in range(len(y)):
        c = m["names"][int(y[i])]
        shown.setdefault(c, []).append(i)
    for c in ("sneaker", "bag", "t_shirt"):
        if c not in shown:
            continue
        for i in shown[c][:4]:
            q = " ".join(d["qvocab"][t] for t in d["Qtr"][i] if t != 0)
            print(f"    image={c:<10} question={q:<22} answer={d['avocab'][d['Atr'][i]]}")
    print()
    # how balanced are the answers?
    u, c = np.unique(d["Atr"][:], return_counts=True)
    print("  answer distribution in this slice:")
    for i, n in zip(u, c):
        print(f"    {d['avocab'][i]:<22} {n:>6}  ({n/len(d['Atr'])*100:.1f}%)")
