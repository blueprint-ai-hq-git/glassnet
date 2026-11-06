#!/bin/bash
# SPLIT-LEAK AUDIT for the OTHER LINES -- the talk leak (36.7% of val shared content with train)
# inflated that line's headline from 54.89% to 80.59%. Every other line's accuracy number is
# equally suspect until its split is checked. Exact-content duplicates across the split boundary:
#   * image (fashion 60k/10k) and medical (PathMNIST 90k/10k/7.2k): hash each 784-pixel image
#   * code (token corpus): hash each 100-token window, like the talk audit
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, hashlib, os

def img_hashes(X, n=None):
    h = {}
    for i in range(len(X) if n is None else min(n, len(X))):
        h.setdefault(hashlib.md5(np.ascontiguousarray(X[i]).tobytes()).hexdigest(), []).append(i)
    return h

def report(name, htr, hva):
    dup = set(htr) & set(hva)
    print(f"  {name:<26} train images hashed {sum(len(v) for v in htr.values()):>6}  "
          f"val {sum(len(v) for v in hva.values()):>6}  identical images across the split: {len(dup)}")
    if dup:
        ex = list(dup)[:2]
        for e in ex:
            print(f"      example: train idx {htr[e][:3]} == val idx {hva[e][:3]}")

z = np.load("data/fashion_mnist_cache.npz")
report("fashion (image line)", img_hashes(z["X_train"]), img_hashes(z["X_test"]))

z = np.load("data/loop_pathmnist_gray.npz")
report("pathmnist (medical)", img_hashes(z["Xtr"]), img_hashes(z["Xval"]))
report("pathmnist tr/te", img_hashes(z["Xtr"]), img_hashes(z["Xte"]))

# code line: token windows
for tag in ("v32000_clean", "v32000_code"):
    ip = f"data/code/prepared/code_ids_{tag}.npy"
    bp = f"data/code/prepared/code_bounds_{tag}.npy"
    spp = f"data/code/prepared/code_split_{tag}.npy"
    if not all(os.path.exists(p) for p in (ip, bp, spp)):
        print(f"  code {tag:<20} missing arrays, skipped"); continue
    ids = np.load(ip); b = np.load(bp); s = np.load(spp)
    tr, va = set(), set()
    for j in range(len(b) - 1):
        t = ids[b[j]:b[j+1]]
        if len(t) < 110:
            continue
        h = hashlib.md5(t[10:110].tobytes()).hexdigest()
        (tr if s[j] == 0 else va).add(h)
    inter = tr & va
    print(f"  code {tag:<20} train windows {len(tr):>6}  val windows {len(va):>6}  "
          f"identical 100-token windows across the split: {len(inter)} "
          f"({len(inter)/max(len(va),1)*100:.1f}% of val)")
PY