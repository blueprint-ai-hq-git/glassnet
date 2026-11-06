"""vl_data.py — image + caption pairs from data that is ALREADY local and licensed.

WHY NOT A WEB CAPTION DATASET
-----------------------------
COCO (HuggingFaceM4) is a loading script, VQAv2/OBELICS are parquet, and reading parquet needs
`pyarrow` -- which is not installed and is not on this project's allowed list. The honest options
were: install a parser, or use data that is already here and can be read with NumPy.

Fashion-MNIST (MIT) and MedMNIST (CC BY 4.0) are both already downloaded and both NumPy-readable.
Their class labels give REAL image->text pairs. A caption template turns a label into a sentence:

    class 7 (sneaker)  ->  "a photo of a sneaker"

That is a genuine captioning task, not a toy: the model must look at pixels and emit the right words
in the right order, and it is CHECKABLE -- the generated caption either names the right class or it
does not.

MEDICAL: MedMNIST is medical imaging under CC BY 4.0 (blood, breast, pneumonia, retina...), which
gives the medical-information angle with a licence that passes the project's rule.
"""
import json
import os

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

FASHION_NAMES = ["t shirt", "trouser", "pullover", "dress", "coat",
                 "sandal", "shirt", "sneaker", "bag", "ankle boot"]

MEDMNIST_NAMES = {
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
                   "lymphocyte", "monocyte", "neutrophil", "platelet"],
    "breastmnist": ["malignant", "normal_benign"],
    "pneumoniamnist": ["normal", "pneumonia"],
    "pathmnist": ["adipose", "background", "debris", "lymphocytes", "mucus",
                  "smooth_muscle", "normal_colon_mucosa", "cancer_associated_stroma",
                  "colorectal_adenocarcinoma"],
    "dermamnist": ["actinic_keratosis", "basal_cell_carcinoma", "benign_keratosis",
                   "dermatofibroma", "melanoma", "nevus", "vascular_lesion"],
    "octmnist": ["choroidal_neovascularization", "diabetic_macular_edema", "drusen", "normal"],
    "tissuemnnist": ["unknown"],
    "retinamnist": ["normal", "mild", "moderate", "severe", "proliferative"],
    "organamnist": ["bladder", "femur_left", "femur_right", "heart", "kidney_left",
                    "kidney_right", "liver", "lung_left", "lung_right", "pancreas",
                    "spleen"],
}

CAPTION_TEMPLATE = "a photo of the {}"

# ------------------------------------------------------------------------------------------------
# WHY QUESTIONS, NOT A FIXED CAPTION PREFIX
# -------------------------------------------------------------------------------------------------
# The first design fed the model a fixed prefix ("a photo of the") and asked it for the class word.
# That makes every text concept CONSTANT -- there is nothing in the prefix to detect -- so the text
# half of the fused layer would have carried no information and "fused" would have been a label
# rather than a fact.
#
# Varied questions fix it. Each question template is a different way of asking for the same thing,
# so:
#   * the TEXT concepts (asks_what / asks_name / asks_see / asks_identify) genuinely vary per sample
#   * the IMAGE concepts (is_bright / left_right_symmetric / ...) genuinely vary per sample
#   * the answer layer must combine BOTH -- "which question is this" AND "what is in the picture" --
#     to produce the right class word.
# That is a real VQA task with an interpretable middle, and it is checkable: the answer either names
# the right class or it does not.
QUESTIONS = [
    ("what is in the photo", "asks_what"),
    ("name the object", "asks_name"),
    ("what do you see", "asks_see"),
    ("identify this item", "asks_identify"),
]
ANSWER_TEMPLATE = "{}"


def build_questions(rng, n):
    """Assign each sample one of the question templates, evenly and reproducibly."""
    qs = []
    for i in range(n):
        qs.append(QUESTIONS[i % len(QUESTIONS)])
    return qs


def question_vocab(qs):
    toks = []
    for q, _ in qs:
        toks.extend(q.split())
    uniq = ["<pad>", "<unk>"] + sorted(set(toks))
    return uniq, {w: i for i, w in enumerate(uniq)}


def encode_question(q, vmap, length):
    ids = [vmap.get(w, 1) for w in q.split()][:length]
    ids = ids + [0] * (length - len(ids))
    return ids


def load_fashion():
    cache = os.path.join(PROJ, "data", "fashion_mnist_cache.npz")
    d = np.load(cache)
    Xtr, ytr = d["X_train"].astype(np.float32), d["y_train"].astype(np.int64)
    Xte, yte = d["X_test"].astype(np.float32), d["y_test"].astype(np.int64)
    if Xtr.max() > 1.5:
        Xtr, Xte = Xtr / 255.0, Xte / 255.0
    return Xtr, ytr, Xte, yte, FASHION_NAMES


def load_medmnist(which="bloodmnist", color=False):
    """MedMNIST npz files: keys are 'train_images','train_labels', etc. 28x28 grayscale.

    `color=True` keeps the RGB CHANNELS instead of averaging them. This matters: PathMNIST's raw
    channel means are 188.8 / 136.0 / 180.0, so averaging destroys the stain colour that separates
    tissue types in histology. The returned layout is CHANNEL-MAJOR -- (B, C*784) meaning
    [C, 28, 28] flattened -- because that is what PatchEmbed reshapes into.
    """
    path = os.path.join(PROJ, "data", "medmnist", f"{which}.npz")
    if not os.path.exists(path):
        for cand in os.listdir(os.path.join(PROJ, "data", "medmnist")):
            if which in cand and cand.endswith(".npz"):
                path = os.path.join(PROJ, "data", "medmnist", cand)
                break
    if not os.path.exists(path):
        return None
    z = np.load(path)
    keys = list(z.keys())

    def grab(split):
        for k in keys:
            if k.startswith(split) and "image" in k:
                imgs = z[k]
                break
        else:
            return None, None
        for k in keys:
            if k.startswith(split) and "label" in k:
                labs = z[k]
                break
        else:
            labs = None
        return imgs, labs

    Xtr, ytr = grab("train")
    Xte, yte = grab("test")
    if Xtr is None:
        return None
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    if Xtr.ndim == 4 and Xtr.shape[-1] == 3:
        if color:
            # channel-major flat: (B, 28, 28, 3) -> (B, 3, 28, 28) -> (B, 3*784)
            Xtr = Xtr.transpose(0, 3, 1, 2).reshape(len(Xtr), -1)
            Xte = Xte.transpose(0, 3, 1, 2).reshape(len(Xte), -1)
        else:
            Xtr = Xtr.mean(axis=-1)
            Xte = Xte.mean(axis=-1)
    if Xtr.ndim == 3 and Xtr.shape[-1] == 28:
        Xtr = Xtr.reshape(len(Xtr), -1)
        Xte = Xte.reshape(len(Xte), -1)
    Xtr, Xte = Xtr / max(Xtr.max(), 1e-6), Xte / max(Xte.max(), 1e-6)
    if Xtr.shape[1] not in (784, 2352):
        return None
    ytr = np.asarray(ytr).reshape(len(Xtr)).astype(np.int64)
    yte = np.asarray(yte).reshape(len(Xte)).astype(np.int64)
    ytr = np.clip(ytr, 0, len(MEDMNIST_NAMES.get(which, ["x"] * 20)) - 1)
    yte = np.clip(yte, 0, len(MEDMNIST_NAMES.get(which, ["x"] * 20)) - 1)
    return Xtr, ytr, Xte, yte, MEDMNIST_NAMES.get(which, ["class"] * 20)


def build_captions(y, names, max_words=6):
    """label -> 'a photo of a <name>'. Names with two words are joined with '_' first."""
    caps = []
    for lab in y:
        nm = names[int(lab)].replace(" ", "_")
        caps.append(CAPTION_TEMPLATE.format(nm))
    return caps


def caption_vocab(caps):
    toks = []
    for c in caps:
        toks.extend(c.split())
    uniq = ["<pad>", "<unk>"] + sorted(set(toks))
    return uniq, {w: i for i, w in enumerate(uniq)}


def encode_caption(cap, vmap, length=6):
    """'a photo of a sneaker' -> [ids..., <pad>...] trimmed to `length`.

    The caption is TRUNCATED at length and the LAST token position is the training target: given the
    image and the first n-1 words, predict word n. So a length-6 caption trains 5 prediction steps
    per image, all conditioned on the picture.
    """
    ids = [vmap.get(w, 1) for w in cap.split()][:length]
    ids = ids + [0] * (length - len(ids))
    return ids


def prepare_vqa(which="fashion", max_train=None, seed=0, color=False):
    """Image + QUESTION -> ANSWER (the class name). A real VQA task with a fused, readable middle.

    `color=True` keeps the RGB channels for RGB datasets instead of averaging them.

    Returns
      Xtr/Xte   images (N, 784) or (N, 2352) channel-major when color
      Qtr/Qte   question token ids (N, qlen)
      Atr/Ate   answer token ids (N, 1)  -- the class name
      ytr/yte   class labels (for the caption-accuracy metric)
    """
    if which == "fashion":
        Xtr, ytr, Xte, yte, names = load_fashion()
    else:
        out = load_medmnist(which, color=color)
        if out is None:
            return None
        Xtr, ytr, Xte, yte, names = out

    qs_tr = build_questions(np.random.default_rng(seed), len(Xtr))
    qs_te = build_questions(np.random.default_rng(seed), len(Xte))
    qvocab, qvmap = question_vocab(QUESTIONS)
    qlen = max(len(q.split()) for q, _ in QUESTIONS)

    # answer = the class name as ONE token. Spaces become underscores so the answer stays one word.
    awords = sorted({names[i].replace(" ", "_") for i in range(len(names))})
    avocab = ["<pad>", "<unk>"] + awords
    avmap = {w: i for i, w in enumerate(avocab)}

    Qtr = np.asarray([encode_question(q, qvmap, qlen) for q, _ in qs_tr], dtype=np.int64)
    Qte = np.asarray([encode_question(q, qvmap, qlen) for q, _ in qs_te], dtype=np.int64)
    Atr = np.asarray([[avmap[names[int(i)].replace(" ", "_")]] for i in ytr], dtype=np.int64)
    Ate = np.asarray([[avmap[names[int(i)].replace(" ", "_")]] for i in yte], dtype=np.int64)

    if max_train and max_train < len(Xtr):
        sel = np.random.default_rng(seed).permutation(len(Xtr))[:max_train]
        Xtr, ytr, Qtr, Atr = Xtr[sel], ytr[sel], Qtr[sel], Atr[sel]

    meta = {"which": which, "names": names, "qlen": qlen, "n_train": len(Xtr), "n_test": len(Xte),
            "source": "Fashion-MNIST (MIT)" if which == "fashion"
                      else f"MedMNIST {which} (CC BY 4.0)",
            "questions": [q for q, _ in QUESTIONS],
            "qconcepts": [c for _, c in QUESTIONS],
            "answer_vocab": avocab}
    return {"Xtr": Xtr, "ytr": ytr, "Qtr": Qtr, "Atr": Atr,
            "Xte": Xte, "yte": yte, "Qte": Qte, "Ate": Ate,
            "qvocab": qvocab, "qvmap": qvmap, "avocab": avocab, "avmap": avmap,
            "meta": meta}


def prepare(which="fashion", max_train=None, seed=0):
    if which == "fashion":
        Xtr, ytr, Xte, yte, names = load_fashion()
    else:
        out = load_medmnist(which)
        if out is None:
            return None
        Xtr, ytr, Xte, yte, names = out
    caps_tr = build_captions(ytr, names)
    caps_te = build_captions(yte, names)
    vocab, vmap = caption_vocab(caps_tr)
    L = max(len(c.split()) for c in caps_tr)
    Ctr = np.asarray([encode_caption(c, vmap, L) for c in caps_tr], dtype=np.int64)
    Cte = np.asarray([encode_caption(c, vmap, L) for c in caps_te], dtype=np.int64)
    if max_train and max_train < len(Xtr):
        sel = np.random.default_rng(seed).permutation(len(Xtr))[:max_train]
        Xtr, ytr, Ctr = Xtr[sel], ytr[sel], Ctr[sel]
        caps_tr = [caps_tr[i] for i in sel]      # keep the strings aligned with the arrays, or a
        #                                          later reader compares a caption to the wrong image
    meta = {"which": which, "names": names, "caption_len": L,
            "vocab_size": len(vocab), "n_train": len(Xtr), "n_test": len(Xte),
            "source": "Fashion-MNIST (MIT)" if which == "fashion" else f"MedMNIST {which} (CC BY 4.0)"}
    return {"Xtr": Xtr, "ytr": ytr, "Ctr": Ctr, "Xte": Xte, "yte": yte, "Cte": Cte,
            "vocab": vocab, "vmap": vmap, "meta": meta,
            "caps_tr": caps_tr, "caps_te": caps_te}


if __name__ == "__main__":
    import numpy as np
    d = prepare("fashion", max_train=2000)
    m = d["meta"]
    vocab = d["vocab"]
    print(f"  {m['source']}")
    print(f"  train {d['Xtr'].shape}  test {d['Xte'].shape}  caption_len {m['caption_len']}")
    print(f"  caption vocab ({m['vocab_size']}): {vocab}")
    print()
    print("  ROUND-TRIP CHECK (decode the ids back to words -- the ids are what the model sees):")
    print(f"  {'label':>6} {'class name':<14} {'decoded from IDS':<30} original string")
    ok = 0
    for i in range(6):
        dec = " ".join(vocab[t] for t in d["Ctr"][i] if t != 0)
        orig = d["caps_tr"][i]
        match = dec == orig
        ok += match
        print(f"  {d['ytr'][i]:>6} {m['names'][d['ytr'][i]]:<14} {dec:<30} {orig!r}  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  -> {ok}/6 round-trip exactly")
    print()
    print("  --- medmnist files present ---")
    md = os.path.join(PROJ, "data", "medmnist")
    for f in sorted(os.listdir(md)):
        p = os.path.join(md, f)
        print(f"    {f:<34} {os.path.getsize(p)/1e6:8.1f} MB")
