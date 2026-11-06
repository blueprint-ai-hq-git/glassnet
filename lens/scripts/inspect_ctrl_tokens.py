
import numpy as np, os, json
d = "/run/media/yahya/HARD DISK 4/GlassNet/data/chat/prepared_mix_ctrl_honest"
files = sorted(os.listdir(d)); print("files:", files)
vocab = json.load(open(os.path.join(d, "vocab.json"))) if os.path.exists(os.path.join(d, "vocab.json")) else None
if vocab is None:
    for cand in ("vocab_v12000.json", "vocab.json", "vocab.txt"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            vocab = json.load(open(p)) if p.endswith(".json") else open(p).read().split("\n")
            print("vocab from", cand); break
ids = np.load(os.path.join(d, "ids_v12000.npy"))
split = np.load(os.path.join(d, "split_v12000.npy")) if os.path.exists(os.path.join(d, "split_v12000.npy")) else None
print("ids", ids.shape, ids.dtype, "split", None if split is None else (split.shape, np.bincount(split)))
va = np.where(split == 1)[0] if split is not None else []
print("val conversations:", len(va))
# what do the FIRST tokens of a val conversation look like?
for j in va[:2]:
    a = int(np.load(os.path.join(d, "bounds_v12000.npy"))[j]) if os.path.exists(os.path.join(d, "bounds_v12000.npy")) else None
    if a is None:
        bounds = np.load(os.path.join(d, "bounds.npy")); a = int(bounds[j])
    seg = ids[a:a+30]
    print("  val conv", j, "->", [vocab[int(t)] for t in seg])
