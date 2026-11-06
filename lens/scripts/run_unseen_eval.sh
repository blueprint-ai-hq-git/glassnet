#!/bin/bash
# THE UNSEEN-UNIQUE YARDSTICK — both talk models measured on val windows whose content has NO
# twin in EITHER train set (the only number fair to both, and the one that decides the dedupe
# question). The raw-trained model memorised 383 val-conv twins (that part of its 56.07% is
# remembered, not learned); the clean-trained model was trained on 38% fewer tokens. Excluding
# the leak marks both down and compares what each did on data neither had seen.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - "$@" <<'PY'
import sys, os
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import numpy as np, json, hashlib
# raw train twins of every window
rids = np.load("data/chat/prepared/ids_v12000.npy")
rb   = np.load("data/chat/prepared/bounds_v12000.npy")
rs   = np.load("data/chat/prepared/split_v12000.npy")
cid  = np.load("data/chat/prepared_clean/ids_v12000.npy")
cb   = np.load("data/chat/prepared_clean/bounds_v12000.npy")
cs   = np.load("data/chat/prepared_clean/split_v12000.npy")
vocab = json.load(open("data/chat/prepared/vocab_v12000.json"))

def wh(j, ids, b):
    t = ids[b[j]:b[j+1]]
    return None if len(t) < 110 else hashlib.md5(t[10:110].tobytes()).hexdigest()

train_hashes = set()
for j in range(len(rb) - 1):
    if rs[j] == 0:
        h = wh(j, rids, rb)
        if h: train_hashes.add(h)
# clean-val convs with a raw-train twin -> EXCLUDED from the yardstick
leaked = 0
for j in range(len(cb) - 1):
    if cs[j] == 1 and wh(j, cid, cb) in train_hashes:
        leaked += 1
print(f"clean-val convs with a RAW-train twin: {leaked} (excluded from the unseen set)")

os.environ["CHAT_DATA_DIR"] = "/run/media/yahya/HARD DISK 4/GlassNet/data/chat/prepared_clean"
import chat_data, lens
WIN = 32
d = chat_data.build()
ids, bounds, split = d["ids"], d["bounds"], d["split"]
# windows for clean-val convs only, skipping leaked ones
keep_conv = {}
for j in range(len(bounds) - 1):
    if split[j] == 1:
        keep_conv[j] = wh(j, ids, bounds) not in train_hashes
import numpy as np
Xva, Yva, Wva = [], [], []
is_asst = np.zeros(len(ids), dtype=np.float32)
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j+1]
    toks = [vocab[i] for i in ids[a0:b0]]
    in_a, k = False, 0
    while k < len(toks):
        if toks[k].lower() == "assistant" and k+1 < len(toks) and toks[k+1] == ":":
            in_a, k = True, k+2; continue
        if toks[k].lower() == "human" and k+1 < len(toks) and toks[k+1] == ":":
            in_a, k = False, k+2; continue
        if in_a: is_asst[a0+k] = 1.0
        k += 1
for j in range(len(bounds) - 1):
    if split[j] != 1 or not keep_conv[j]: continue
    s0, s1 = bounds[j], bounds[j+1]
    for s in range(s0, max(s1 - WIN, s0)):
        Xva.append(ids[s:s+WIN]); Yva.append(ids[s+WIN]); Wva.append(is_asst[s+WIN])
Xva = np.asarray(Xva); Yva = np.asarray(Yva); Wva = np.asarray(Wva)
print(f"unseen-unique val: {len(Xva)} windows, {int((Wva>0.5).sum())} assistant targets")

def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)
def evaluate(net, mask=True):
    P, C, S = [], 0, 0
    for s in range(0, len(Xva), 512):
        idx = np.arange(s, min(s+512, len(Xva)))
        p = softmax(net.forward_arrays(Xva[idx])["logits"]); t = Yva[idx]
        if mask:
            m = Wva[idx] > 0.5
            if m.sum() == 0: continue
            p, t = p[m], t[m]
        P.append(-np.log(np.maximum(p[np.arange(len(t)), t], 1e-12)))
        C += int((p.argmax(axis=1) == t).sum()); S += len(t)
    P = np.concatenate(P)
    return float(np.exp(P.mean())), C / max(S, 1), S
for name, ship in (("t11 d256 (raw data)", "lens/runs/chat/t11_talk41_ep192.npz"),
                   ("t12 d512 (raw data)", "lens/runs/chat/t12_d512_full_ep192.npz"),
                   ("t12 d512 (clean data)", "lens/runs/chat/t12_clean_d512_ep192.npz")):
    net = lens.load(os.path.join("/run/media/yahya/HARD DISK 4/GlassNet", ship), device="gpu")
    mp, ma, ms = evaluate(net, True)
    print(f"  {name:<24} unseen-unique masked acc {ma*100:6.2f}%  ppl {mp:8.1f}  (n={ms})")
PY