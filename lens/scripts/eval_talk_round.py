"""eval_talk_round.py — generic talk-round gate: PASS iff
  (a) unk rate on shared probes <= 1%        (no collapse)
  (b) last-epoch val_acc >= twin + 2.0 pts   (same margin rule as T1: the pre-registered bar)
  (c) generation still reads as multi-word fluent prose (>=3 words on every probe)
USAGE: eval_talk_round.py <new_ckpt.npz> <new_log.json> <twin_log.json> [tag]
Verdict BINARY. exit 0 = PASS, 2 = FAIL (gate not moved), 3 = ERROR (chain must stop).
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import variants  # noqa: E402

new_ckpt, new_log, twin_log = sys.argv[1], sys.argv[2], sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(new_ckpt).replace(".npz", "")
BAR = 2.0  # the standing margin rule: twin + 2.0 pts

nl = json.load(open(new_log))
tl = json.load(open(twin_log))
n_acc = nl["history"][-1]["val_acc"] * 100
t_acc = tl["history"][-1]["val_acc"] * 100
n_ppl = nl["history"][-1]["val_ppl"]
bar = t_acc + BAR

cv = json.load(open(os.path.join(PROJ, "data", "chat", "prepared", variants.CHAT_VOCAB_FILE)))
v = variants.Variants(talk=new_ckpt, chat_vocab=cv, device="cpu")
PROBES = ["What is 2+2?", "Hello", "The weather today is", "Tell me a story about a dog",
          "How are you doing?", "What time is it?"]
unks = toks = 0
gens = []
for q in PROBES:
    r = v.talk_sentence(q, n_new=25)
    rep = str(r["reply"] if isinstance(r, dict) else r)
    unks += rep.count("<unk>"); toks += len(rep.split())
    gens.append(rep[:58])
unk_rate = unks / max(toks, 1) * 100

print("=" * 100)
print(f"TALK ROUND GATE [{tag}]   twin={os.path.basename(twin_log)}  bar={bar:.2f}%")
print(f"  val_acc {n_acc:.2f}% (twin {t_acc:.2f}%)   val_ppl {n_ppl:.1f}   unk {unk_rate:.2f}%")
for g in gens:
    print(f"    {g}")
ok_a = unk_rate <= 1.0
ok_b = n_acc >= bar
ok_c = all(len(g.split()) >= 3 for g in gens)
print(f"  a unk<=1%      -> {'PASS' if ok_a else 'FAIL'}  ({unk_rate:.2f}%)")
print(f"  b val>=bar     -> {'PASS' if ok_b else 'FAIL'}  ({n_acc:.2f} >= {bar:.2f})")
print(f"  c fluent prose -> {'PASS' if ok_c else 'FAIL'}")
if ok_a and ok_b and ok_c:
    print(f"  VERDICT: [{tag}] PASS -> candidate for shipping talk")
    sys.exit(0)
print(f"  VERDICT: [{tag}] FAIL -> this lever was not the blocker; gate NOT moved")
sys.exit(2)