"""why_val_mask_probe2.py — walk the val conversations EXACTLY as train_chat.py does and count."""
import json
import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
DD = P + "/data/chat/prepared_mix_ctrl_honest"
vocab = json.load(open(DD + "/vocab_v12000.json"))
bounds = np.load(DD + "/bounds_v12000.npy")
split = np.load(DD + "/split_v12000.npy").astype(np.int8)
ids = np.load(DD + "/ids_v12000.npy", mmap_mode="r")

is_asst = np.zeros(len(ids), dtype=bool)
val = np.where(split == 1)[0][:500]
for j in val:
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    k = 0
    in_asst = False
    while k < len(toks):
        if toks[k].lower() == "<|im_start|>" and k + 1 < len(toks):
            in_asst = toks[k + 1].lower() == "assistant"
            k += 2
            continue
        if toks[k].lower() == "<|im_end|>":
            k += 1
            continue
        if in_asst:
            is_asst[a0 + k] = True
        k += 1
print(f"marked assistant positions in the first {len(val)} val conversations: {int(is_asst.sum()):,}")
tot = sum(int(bounds[j + 1]) - int(bounds[j]) for j in val)
print(f"total tokens in those conversations: {tot:,}  ratio {is_asst.sum()/max(tot,1):.3f}")

# now the targets the trainer masks on
import sys
sys.path.insert(0, P + "/lens/scripts")
import chat_data
Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, 32, which=1, return_index=True)
m = is_asst[Yva]
print(f"val windows {len(Xva):,}   masked targets {int(m.sum()):,} ({m.mean()*100:.1f}%)")
print(f"first 12 val window starts {Sva[:12]}  targets {Yva[:12]}")
print("first val window tokens:", " ".join(vocab[i] for i in np.asarray(Xva[0])))
print("its target token      :", vocab[int(Yva[0])])
print(f"is_asst over the whole corpus: {int(is_asst.sum()):,}")
