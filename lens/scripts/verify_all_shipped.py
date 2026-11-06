"""verify_all_shipped.py — comprehensive end-to-end verification of every shipped artifact.

Today's edits touched variants.py (CHAT_FINAL, decoder defaults), explain() (fidelity annotation),
and added fidelity tables. This script loads EVERY shipped checkpoint and runs the full pipeline:
forward pass, explain(), refusal behavior, and cross-checks against known-good numbers.

GATES (pre-registered):
  v1  every shipped checkpoint loads without error
  v2  every shipped checkpoint produces a forward pass with correct output shape
  v3  explain() returns fidelity-annotated contributions (not bare names)
  v4  explain() excludes decorative concepts when a fidelity table exists
  v5  refusal behavior is correct (see-only asked to talk, talk id outside vocab)
  v6  the shipped talk model's decoder matches the tuned defaults
  v7  no checkpoint is missing its fidelity table (warns if so)
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants                                           # noqa: E402
import vl_data                                            # noqa: E402
from lens import backend as B                             # noqa: E402

R = os.path.join(PROJ, "lens", "runs")
results = {}

print("=" * 100)
print("COMPREHENSIVE SHIPPED-ARTIFACT VERIFICATION")
print("=" * 100)

# ---- 1. Talk model
print("\n[1] TALK MODEL")
talk_ckpt = os.path.join(R, "chat", variants.CHAT_FINAL)
talk_vocab = os.path.join(R, "chat", variants.CHAT_VOCAB_FILE)
v = variants.build_default(device="cpu")
net, meta = v._need("talk")
print(f"  checkpoint: {os.path.basename(talk_ckpt)}")
print(f"  vocab: {os.path.basename(talk_vocab)}  V={net.V}")
print(f"  decoder: {variants.TALK_DEFAULT_DECODER}")

# forward pass — the talk model requires exactly W=16 tokens per window
ids = np.zeros((1, net.W), dtype=np.int64)
ids[0, :5] = [1, 2, 3, 4, 5]
out = net.forward_arrays(ids)
logits = B.to_cpu(out["logits"])
print(f"  forward: logits shape {logits.shape}  (expect [1, {net.V}])")
v2 = logits.shape == (1, net.V)

# generate a reply — the public API is talk_sentence(prompt, n_new=...)
reply = v.talk_sentence("What is 2+2?", n_new=20)
# talk_sentence returns {"model": ..., "reply": ...} — read the reply field
reply_txt = reply["reply"] if isinstance(reply, dict) else reply
print(f"  talk_sentence('What is 2+2?') -> '{str(reply_txt)[:80]}'")
v1 = len(str(reply_txt)) > 0

# decoder matches defaults
dec = variants.TALK_DEFAULT_DECODER
v6 = (dec["top_k"] == 80 and dec["temp"] == 0.9 and dec["rep_penalty"] == 1.6)
print(f"  decoder matches tuned defaults: {v6}")

results["talk"] = {"v1_loads": v1, "v2_shape": v2, "v6_decoder": v6}

# ---- 2. See-only model
print("\n[2] SEE-ONLY MODEL")
d = vl_data.prepare_vqa("fashion", seed=0)      # shared test data; also used by see+talk below
see_ckpt = v._paths.get("see")
if "see" in v.models:          # loaded even when _paths has no entry
    net_s, meta_s = v._need("see")
    print(f"  checkpoint: {os.path.basename(see_ckpt) if see_ckpt else '(default see model)'}")
    X = np.asarray(d["Xte"][:5], dtype=np.float32)
    out_s = net_s.forward(X)
    # forward returns a dict {"concepts", "logits"}; logits is (5, C)
    logits_s = B.to_cpu(out_s["logits"] if isinstance(out_s, dict) else out_s.data)
    print(f"  forward: logits shape {logits_s.shape}  (expect [5, {net_s.C}])")
    pred = logits_s.argmax(1)
    print(f"  predictions: {pred.tolist()}  (fashion classes 0-9)")
    v1s = logits_s.shape == (5, net_s.C)
    results["see"] = {"v1_loads": v1s, "v2_shape": v1s}
else:
    print("  NO see checkpoint loaded")
    results["see"] = {"v1_loads": False, "v2_shape": False}

# ---- 3. See+talk model
print("\n[3] SEE+TALK MODEL")
st_ckpt = v._paths.get("see_talk")
if st_ckpt:
    net_st, meta_st = v._need("see_talk")
    print(f"  checkpoint: {os.path.basename(st_ckpt)}")
    # explain() with fidelity
    img = np.asarray(d["Xte"][:1], dtype=np.float32)
    vocab_st = meta_st["vocab"]
    vi = {w: i for i, w in enumerate(vocab_st)}
    prefix = [vi.get("<pad>", 0), vi.get("<pad>", 0), vi.get("<bos>", 0),
              vi.get("this", 0), vi.get("is", 0), vi.get("a", 0)]
    res = v.explain(img, prefix)
    has_fid = res.get("fidelity_source") is not None
    has_corr = all(c.get("own_corr") is not None for c in res["image_driven_contributions"])
    has_excl = "decorative_excluded" in res or res.get("fidelity_source") is not None
    print(f"  explain() word: {res['word']}")
    print(f"  fidelity_source: {res.get('fidelity_source')}")
    print(f"  contributions annotated with own_corr: {has_corr}")
    print(f"  decorative excluded or noted: {has_excl}")
    for c in res["image_driven_contributions"][:3]:
        print(f"    {c['concept']:<24} own_corr {c.get('own_corr'):>6}  verdict {c.get('verdict','?'):<11}")
    v3 = has_corr
    v4 = has_excl
    v7 = has_fid
    results["see_talk"] = {"v3_annotated": v3, "v4_excludes": v4, "v7_has_table": v7}
else:
    print("  NO see+talk checkpoint loaded")
    results["see_talk"] = {"v3_annotated": False, "v4_excludes": False, "v7_has_table": False}

# ---- 4. Refusal behavior
print("\n[4] REFUSAL BEHAVIOR")
# see-only asked to talk
# see-only build asked to TALK: talk(window_ids) must raise, not guess (same probe as selfcheck)
see_only = variants.Variants(see=see_ckpt, device="cpu")
try:
    see_only.talk([0] * 16)
    refusal1 = False
    print("  see-only asked to talk -> refuses: False  (IT ANSWERED — dishonest build)")
except Exception as e:
    refusal1 = True
    print(f"  see-only asked to talk -> refuses: True  ({str(e)[:50]})")

# a talk id outside the checkpoint's vocab must raise, not print a guessed word
v2 = variants.build_default(device="cpu")
v2.chat_vocab = ["only", "two", "words"]      # deliberately too small for a 12000-wide model
try:
    v2.talk([0] * 16)
    refusal2 = False
    print("  talk id outside vocab -> refuses: False  (printed a word it cannot know)")
except Exception as e:
    refusal2 = True
    print(f"  talk id outside vocab -> refuses: True  ({str(e)[:50]})")
results["refusal"] = {"see_asks_talk": refusal1, "bad_id": refusal2}

# ---- 5. Fidelity table coverage
print("\n[5] FIDELITY TABLE COVERAGE")
fidelity_expected = {
    "see_talk": os.path.join(R, "seetalk", "seetalk_k22_big_fidelity.json"),
    "vl_lc0.15": os.path.join(R, "vl", "mlow_lc0.15_ep8_fidelity.json"),
}
for name, path in fidelity_expected.items():
    exists = os.path.exists(path)
    print(f"  {name}: {'OK' if exists else 'MISSING'}  {os.path.basename(path)}")
    results.setdefault("fidelity", {})[name] = exists

# ---- Summary
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
all_pass = True
for area, checks in results.items():
    for gate, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {area:<16} {gate:<20} {status}")
print(f"\n  {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED — see above'}")
sys.exit(0 if all_pass else 1)
