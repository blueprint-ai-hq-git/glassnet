"""Which vocab tokens does lens.numeric call a NUMBER with a non-finite value? (bug probe, read-only)"""
import json, os, sys, numpy as np
PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
os.chdir(PROJ)
from lens.numeric import token_value, _is_number

D = "data/chat/prepared_mix_ctrl_honest"
vocab = json.load(open(D + "/vocab_v12000.json"))
bad = []
for i, t in enumerate(vocab):
    if _is_number(t):
        v = token_value(t)
        if not np.isfinite(v):
            bad.append((i, t, v))
print("numeric tokens whose value is NOT finite:", len(bad))
for b in bad[:20]:
    print("  id", b[0], "token", repr(b[1]), "value", b[2])
n_bad = [t for t in vocab if not _is_number(t) and t.strip().lower() in ("inf", "nan", "infinity")]
print("tokens that LOOK like inf/nan but are not numeric:", n_bad[:10])
# how many tokens in the whole corpus are these?
ids = np.load(D + "/ids_v12000.npy", mmap_mode="r")
if bad:
    ids_bad = np.array([b[0] for b in bad], dtype=np.int64)
    cnt = 0
    for s in range(0, ids.shape[0], 20_000_000):
        chunk = np.asarray(ids[s:s + 20_000_000])
        cnt += int(np.isin(chunk, ids_bad).sum())
    print("occurrences of those tokens in the whole corpus:", cnt)
