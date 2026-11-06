"""eval_t1_window.py — T1 gate: does window 16 -> 32 (ONE change) actually help the talk model?

Pre-registered BEFORE the run (gates fixed):
  t1a  unk rate on the shared probes stays <= 1%   (no collapse; the fluent-filler shape survives)
  t1b  last-epoch val_acc >= 15.4%                  (2+ pts over the W=16 twin's 13.38%)
  t1c  generation still reads as fluent English-ish (the W=16 twin is the reference; probes shown)
Verdict BINARY: ALL THREE gates pass -> T1 wins, promote to shipping talk and merge into see+talk.
Else -> window was not the blocker; report exactly which gate failed with numbers.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants  # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")
TWIN = os.path.join(R, "asonly_v12000_fixedgrad_ep24.npz")
NEW = os.path.join(R, "t1_w32_ep24_ep24.npz")

print("=" * 100)
print("T1 GATE — window 16 -> 32 on the talk line (one change; d=128, vocab 12k, asonly)")
print("=" * 100)

cv = json.load(open(os.path.join(PROJ, "data", "chat", "prepared", variants.CHAT_VOCAB_FILE)))
v_twin = variants.Variants(talk=TWIN, chat_vocab=cv, device="cpu")
v_new = variants.Variants(talk=NEW, chat_vocab=cv, device="cpu")
print(f"  twin: {os.path.basename(TWIN)}")
print(f"  new : {os.path.basename(NEW)}")

# --- gate t1b: last-epoch val_acc from the training logs ---
for tag, path in (("twin(W=16)", TWIN), ("new (W=32)", NEW)):
    log = path.replace(".npz", "_log.json")
    if not os.path.exists(log):
        log = os.path.join(R, "t1_w32_ep24_log.json") if "t1_w32" in path else ""
    if log and os.path.exists(log):
        d = json.load(open(log))
        last = d["history"][-1]
        print(f"  {tag:<13} last-epoch val_acc {last['val_acc']*100:.2f}%  val_ppl {last['val_ppl']:.1f}  "
              f"train_acc {last['train_acc']*100:.2f}%")

# --- gates t1a + t1c: generation on the shared probes ---
PROBES = ["What is 2+2?", "Hello", "The weather today is", "Tell me a story about a dog",
          "How are you doing?", "What time is it?"]

print("\n  generation (n_new=25, same seeds)  unk rate counted over all probes:")
rows = {}
for tag, v in (("twin", v_twin), ("new", v_new)):
    unks = toks = 0
    gen = []
    for q in PROBES:
        r = v.talk_sentence(q, n_new=25)
        rep = r["reply"] if isinstance(r, dict) else r
        rep = str(rep)
        unks += rep.count("<unk>")
        toks += len(rep.split())
        gen.append(rep[:58])
    rate = unks / max(toks, 1) * 100
    rows[tag] = (rate, gen)
    print(f"\n  [{tag}] unk tokens {unks}/{toks} = {rate:.2f}%")
    for g in gen:
        print(f"      {g[:58]}")

# --- verdict ---
new_log = os.path.join(R, "t1_w32_ep24_log.json") if not os.path.exists(NEW.replace(".npz","_log.json")) else NEW.replace(".npz","_log.json")
twin_log = TWIN.replace(".npz", "_log.json")
nlast = json.load(open(new_log))["history"][-1] if os.path.exists(new_log) else {}
tlast = json.load(open(twin_log))["history"][-1] if os.path.exists(twin_log) else {}
n_acc = nlast.get("val_acc", 0) * 100
t_acc = tlast.get("val_acc", 0) * 100
n_unk = rows["new"][0]
t1a = n_unk <= 1.0
t1b = n_acc >= 15.4
t1c_new = rows["new"][1]
# t1c passes if the new model still emits multi-word natural-ish tokens (no <unk> spam, not blank)
t1c = all(len(g.split()) >= 3 for g in t1c_new) and n_unk <= 1.0

print("\n" + "-" * 100)
print(f"  t1a  unk <= 1%                 -> {'PASS' if t1a else 'FAIL'}  ({n_unk:.2f}%)")
print(f"  t1b  val_acc >= 15.4%          -> {'PASS' if t1b else 'FAIL'}  ({n_acc:.2f}% vs twin {t_acc:.2f}%)")
print(f"  t1c  fluent multi-token prose  -> {'PASS' if t1c else 'FAIL'}")
ok = t1a and t1b and t1c
print(f"\n  VERDICT: {'T1 WINS — window was part of the blocker; candidate for shipping talk' if ok else 'T1 FAILS — window was NOT the blocker; next single change is width (d 128 -> 256) or data. Gate not moved.'}")