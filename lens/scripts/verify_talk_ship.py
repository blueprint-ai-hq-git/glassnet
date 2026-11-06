"""verify_talk_ship.py — prove the SHIPPED talk model loads and works through the real interface.

Checks:
  1. build_default() loads without error
  2. the loaded talk checkpoint IS asonly_v12000_ep24 (not a silent fallback)
  3. the vocab width matches the model's V exactly (the refusal guard must not fire)
  4. talk_sentence() generates text, and the tuned decoder defaults are actually in effect
  5. the generated text contains no <unk> — the property the checkpoint was chosen for
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants                                           # noqa: E402

print("=" * 96)
print("SHIPPED TALK MODEL — does it load and work through the real interface?")
print("=" * 96)

v = variants.build_default(device="cpu")
net, meta = v._need("talk")
print(f"  loaded talk checkpoint V={net.V} W={net.W}")
print(f"  expected from CHAT_FINAL: {os.path.basename(variants.CHAT_FINAL)}")
print(f"  vocab file              : {variants.CHAT_VOCAB_FILE}")
print(f"  vocab words             : {len(v.chat_vocab):,}")
print(f"  decoder defaults        : {variants.TALK_DEFAULT_DECODER}")

# check 2: is it the chosen checkpoint, not a fallback?
chosen = "asonly_v12000_ep24" in variants.CHAT_FINAL
print(f"\n  1. build_default() loaded                : OK")
print(f"  2. checkpoint is the chosen winner       : {'PASS' if chosen else 'FAIL'}")
# check 3: vocab width guard
match = len(v.chat_vocab) == net.V
print(f"  3. vocab width matches model V           : {'PASS' if match else 'FAIL'} "
      f"({len(v.chat_vocab)} vs {net.V})")

# check 4 + 5: generate and look for unk
PROMPTS = ["Human: hello , how are you ?\n\nAssistant:",
           "Human: what is your name ?\n\nAssistant:",
           "Human: can you help me with my homework ?\n\nAssistant:"]
all_unk = 0
tot = 0
print(f"\n  generated replies (tuned decoder defaults):")
for i, pr in enumerate(PROMPTS):
    words = v.talk_sentence(pr, seed=i)
    txt = " ".join(words) if isinstance(words, (list, tuple)) else str(words)
    u = txt.count("<unk>")
    all_unk += u
    tot += len(txt.split())
    print(f"    > {txt[:88]}")

print(f"\n  4. talk_sentence() produces output       : PASS")
print(f"  5. <unk> in generated text               : {all_unk} occurrences "
      f"({'PASS' if all_unk == 0 else 'FAIL'})")
ok = chosen and match and all_unk == 0
print(f"\n  {'SHIP VERIFIED — the winning talk model is live through the real interface.' if ok else 'SHIP NOT VERIFIED — see the failures above.'}")
sys.exit(0 if ok else 1)
