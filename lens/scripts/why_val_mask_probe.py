"""why_val_mask_probe.py — the assistant mask came out EMPTY on my held-out walk (self-wake 17:3x).
Nothing downstream of it can be reported until this is explained, so it gets its own tiny probe.
"""
import json
import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
DD = P + "/data/chat/prepared_mix_ctrl_honest"
vocab = json.load(open(DD + "/vocab_v12000.json"))
bounds = np.load(DD + "/bounds_v12000.npy")
split = np.load(DD + "/split_v12000.npy")
ids = np.load(DD + "/ids_v12000.npy", mmap_mode="r")
print(f"tokens {len(ids):,}  convs {len(bounds)-1:,}  split values {np.unique(split, return_counts=True)}")
val_idx = np.where(split == 1)[0]
print(f"val conversations: {len(val_idx):,}   first five: {val_idx[:5]}")
for j in list(val_idx[:3]) + list(np.where(split == 0)[0][:2]):
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    print(f"\nconv {j} (split={split[j]}) {b0-a0} tokens:")
    print("   " + " ".join(toks[:45]))
    print(f"   has <|im_start|>: {toks.count('<|im_start|>')}   has 'assistant': {toks.count('assistant')}")
