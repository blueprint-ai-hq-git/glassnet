"""mask_index_bug_check.py — IS THE ASSISTANT-ONLY MASK INDEXED BY TOKEN ID INSTEAD OF POSITION?

HOW I GOT HERE (2026-10-22 17:5x self-wake, CPU): I rebuilt the trainer's val mask for the talk test and
it came out EMPTY (0 of 87,035 windows). The walker itself works (lens/runs/VAL-MASK-PROBE2.txt: 31,711
positions marked, 72.8% of the tokens it walked). The mismatch is the INDEX EXPRESSION:

    train_chat.py:  Wtr = is_asst[Ytr]        and   Wva = is_asst[Yva] > 0
    Ytr / Yva are TOKEN IDS (chat_data.py: Y = ids_arr[starts + window]), values 0..12,001.
    is_asst is a POSITIONAL bool array over the 294,225,090 corpus tokens.
    The comment above the mask code says the intent is positional: "A window's target is assistant
    text if the flag is ON at that token's position."

So the mask may be reading the first 12,002 CORPUS POSITIONS (used as if they were vocabulary
indices) instead of the position of each target token. This script decides that with numbers.

WHAT THIS CHANGES IF IT IS TRUE: every "MASKED held-out" number in every arm (27.37% ... 35.82%) is
accuracy on a FIXED half of the vocabulary chosen by the first ~12k tokens of the corpus - and
"best checkpoint" was selected by that column. This is the verdict-check bug class: a right number
printed beside a wrong conclusion.
"""
import json
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
DD = P + "/data/chat/prepared_mix_ctrl_honest"
sys.path.insert(0, P + "/lens/scripts")
sys.path.insert(0, P)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import chat_data                                            # noqa: E402
from lens import backend as B                               # noqa: E402
from lens.concepts import Concept                           # noqa: E402
from lens.model import LensNet                              # noqa: E402

W = 32
CKPT = P + "/lens/runs/chat/ct_d256_6M_6ep_best.npz"
LOGGED = 0.3582

vocab = json.load(open(DD + "/vocab_v12000.json"))
bounds = np.load(DD + "/bounds_v12000.npy")
split = np.load(DD + "/split_v12000.npy").astype(np.int8)
ids = np.load(DD + "/ids_v12000.npy", mmap_mode="r")
V = len(vocab)

Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)
print(f"val windows {len(Xva):,}   window {W}   vocab {V:,}   corpus {len(ids):,} tokens")

# ---- 1. the walker, exactly as train_chat.py runs it (all conversations, control-token branch)
is_asst = np.zeros(len(ids), dtype=bool)
is_turnend = np.zeros(len(ids), dtype=bool)
in_asst = False
pos = 0
for j in range(len(bounds) - 1):
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    k = 0
    while k < len(toks):
        if toks[k].lower() == "<|im_start|>" and k + 1 < len(toks):
            in_asst = toks[k + 1].lower() == "assistant"
            k += 2
            continue
        if toks[k].lower() == "<|im_end|>":
            if in_asst:
                is_turnend[a0 + k] = True
            k += 1
            continue
        if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = True
            k += 2
            continue
        if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = False
            k += 2
            continue
        if in_asst:
            is_asst[a0 + k] = True
        k += 1
print(f"is_asst: {int(is_asst.sum()):,} of {len(is_asst):,} corpus POSITIONS marked "
      f"({is_asst.mean()*100:.1f}%)")

# ---- 2. what the trainer's index expression actually reads
print(f"max target token id in the held-out set: {int(Yva.max())}   (vocab size {V:,})")
print(f"max token id anywhere in the corpus    : {int(np.asarray(ids[:2_000_000]).max())}")
head = is_asst[:V]
print(f"is_asst[Yva] can only ever read the first {V:,} entries: {int(head.sum()):,} of them are True "
      f"({head.mean()*100:.1f}%) <- this is the whole content of the mask")

mask_id = is_asst[Yva]                                  # what train_chat.py computes
mask_pos = is_asst[np.asarray(Sva) + W]                 # what the comment says it means
print("")
print(f"MASK BY TOKEN ID (train_chat.py)     : {int(mask_id.sum()):,} of {len(mask_id):,} "
      f"({mask_id.mean()*100:.1f}%)   <- the 'MASKED' column every arm reported")
print(f"MASK BY POSITION (the stated intent): {int(mask_pos.sum()):,} of {len(mask_pos):,} "
      f"({mask_pos.mean()*100:.1f}%)")
both = int((mask_id & mask_pos).sum())
only_id = int((mask_id & ~mask_pos).sum())
only_pos = int((~mask_id & mask_pos).sum())
print(f"agreement: both {both:,}   only-by-id {only_id:,}   only-by-position {only_pos:,}   "
      f"neither {int((~mask_id & ~mask_pos).sum()):,}")
print(f"-> the two masks pick DIFFERENT windows ({100*both/max(len(mask_id),1):.1f}% overlap)")

# a decisive sanity check: does the by-id mask depend on the TARGET ID rather than the window?
targets_in = sorted(set(int(t) for t in Yva[mask_id][:2000]))
print(f"   target ids selected by the by-id mask (first 2000): {targets_in[:15]} ... "
      f"all below {V}: {max(targets_in) < V if targets_in else 'n/a'}")

# ---- 3. the two numbers on the same checkpoint
z = np.load(CKPT, allow_pickle=False)
meta = json.loads(str(z["__meta__"]))
concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
net = LensNet(meta["V"], concepts, d=meta["D"], window=meta["W"], seed=meta["seed"],
              lam_concept=meta["lam_concept"], lam_act=meta["lam_act"],
              attention=bool(meta.get("attention", False)), numeric=bool(meta.get("numeric", False)),
              direct=bool(meta.get("direct", False)), direct_rank=int(meta.get("direct_rank", 16)),
              direct_mode=str(meta.get("direct_mode", "lowrank")))
net.device_name = "cpu"
for p in net.parameters():
    p.data = B.to_dev(z[p.name].astype(np.float32))

lg_all = np.empty((len(Xva), meta["V"]), dtype=np.float32)
for s in range(0, len(Xva), 512):
    sl = slice(s, min(s + 512, len(Xva)))
    lg_all[sl] = net.forward_arrays(np.ascontiguousarray(Xva[sl]))["logits"]
pred = lg_all.argmax(axis=1)
hit = (pred == Yva)


def acc(m):
    return float(hit[m].mean()) if m.sum() else float("nan")


print("")
print(f"checkpoint {CKPT.split('/')[-1]}  (logged MASKED held-out {LOGGED*100:.2f}%)")
print(f"  accuracy on the whole held-out set            {acc(np.ones(len(Yva), bool))*100:.2f}%  (the unmasked column)")
print(f"  accuracy MASK BY TOKEN ID   (what was logged) {acc(mask_id)*100:.2f}%")
print(f"  accuracy MASK BY POSITION   (assistant turns) {acc(mask_pos)*100:.2f}%")
if mask_id.sum() and mask_pos.sum():
    print(f"  accuracy on the windows the by-id mask picked but that are NOT assistant text: "
          f"{acc(mask_id & ~mask_pos)*100:.2f}%  (n={int((mask_id & ~mask_pos).sum()):,})")
    print(f"  accuracy on real assistant windows the by-id mask MISSED: "
          f"{acc(~mask_id & mask_pos)*100:.2f}%  (n={int((~mask_id & mask_pos).sum()):,})")
# the frequency control: is the by-position set simply easier?
for name, m in (("by-id set", mask_id), ("by-position set", mask_pos)):
    ids_sel = Yva[m]
    print(f"  {name}: top-1 accuracy while only ever predicting the 50 most frequent tokens "
          f"{float(np.isin(pred[m], np.argsort(np.bincount(Yva[m], minlength=V))[-50:]).mean())*100:.1f}%")

# what the by-id mask is actually made of
print("")
print("the first 12,002 corpus POSITIONS the by-id mask reads (they are the start of conversation 0):")
print("  " + " ".join(vocab[i] for i in np.asarray(ids[:60])))
print(f"  assistant-marked within that prefix: {int(is_asst[:V].sum()):,} positions")
print("")
print("VERDICT INPUTS ABOVE ARE NUMBERS ONLY. Nothing in this file claims the model talks or does not.")
