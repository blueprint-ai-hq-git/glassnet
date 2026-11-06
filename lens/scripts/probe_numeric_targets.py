"""Probe: how much numeric target is there in the REAL chat corpus, and what values?
Read-only. Prints the distribution used to choose the magnitude SCALE."""
import json, os, sys
import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
D = os.environ.get("CHAT_DATA_DIR") or os.path.join(PROJ, "data/chat/prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D
import chat_data  # noqa
from lens.numeric import token_value, _is_number

ids = np.load(D + "/ids_v12000.npy", mmap_mode="r")
vocab = json.load(open(D + "/vocab_v12000.json"))
bounds = np.load(D + "/bounds_v12000.npy")
split = np.load(D + "/split_v12000.npy")
print("corpus", D)
print("tokens", ids.shape[0], "conversations", bounds.shape[0])

isnum = np.array([_is_number(t) for t in vocab])
val = np.array([token_value(t) for t in vocab], dtype=np.float64)
print("numeric vocab entries:", int(isnum.sum()), "of", len(vocab))

for which, name in ((0, "train"), (1, "val")):
    X, Y = chat_data.make_windows(ids, bounds.tolist(), split.tolist(), 32, which=which,
                                  max_windows=200000, seed=0)
    m = isnum[Y]
    v = val[Y][m]
    print(f"  {name}: windows {len(Y):,}  numeric-target windows {int(m.sum()):,} "
          f"({100.0*m.mean():.2f}%)")
    if len(v):
        av = np.abs(v)
        print(f"    values: min {v.min():.4g} max {v.max():.4g} mean {v.mean():.4g} "
              f"median {np.median(v):.4g}")
        for q in (50, 75, 90, 95, 99, 99.9):
            print(f"      p{q} |v| = {np.percentile(av, q):.4g}")
        print("    |v|<=1:", f"{100.0*np.mean(av<=1):.2f}%", " |v|<=10:",
              f"{100.0*np.mean(av<=10):.2f}%", " |v|<=100:", f"{100.0*np.mean(av<=100):.2f}%",
              " |v|<=1000:", f"{100.0*np.mean(av<=1000):.2f}%")
        print("    MAE of predict-train-mean baseline would be:",
              f"{np.mean(np.abs(v - v.mean())):.4g}", " (val only, upper bound of the baseline)")
