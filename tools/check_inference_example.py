"""check_inference_example.py - gate on examples/inference_numpy.py.

Answers three questions with printed numbers, exit 1 if any fails:
  1. Does the standalone numpy forward equal lens.api.load(...).forward?
  2. Does the example's greedy line reproduce the trainer's own recorded GREEDY line
     for the same prompt (lens/runs/GPU-BEST.txt)?
  3. Does the example refuse a vocabulary that does not belong to the checkpoint?
"""
import json
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "examples")
from inference_numpy import GlassNet, DEFAULT_CKPT, DEFAULT_VOCAB  # noqa: E402
import lens.api as api  # noqa: E402

CKPT = DEFAULT_CKPT
ok = True

z = np.load(CKPT, allow_pickle=False)
meta = json.loads(str(z["__meta__"]))
print("checkpoint %s: V=%s D=%s W=%s direct=%s mode=%s"
      % (CKPT, meta["V"], meta["D"], meta["W"], meta.get("direct"), meta.get("direct_mode")))

m = GlassNet(CKPT, DEFAULT_VOCAB)
net = api.load(CKPT, device="cpu")

# ---- 1. logits equivalence
rng = np.random.default_rng(12345)
ids = rng.integers(0, m.V, size=(8, m.W)).astype(np.int64)
mine = np.stack([m.logits(r)[0] for r in ids])
theirs = api.backend.to_cpu(net.forward(ids)["logits"].data)
mx = float(np.abs(mine - theirs).max())
agree = int((mine.argmax(1) == theirs.argmax(1)).sum())
print("1) max |example - lens.model| logit difference = %.3e over %d windows; argmax %d/%d"
      % (mx, len(ids), agree, len(ids)))
if not (mx < 1e-4 and agree == len(ids)):
    ok = False
    print("   FAIL: the example does not reproduce the model")

# ---- 2. greedy line vs the trainer's recorded line
prompt = "Human: hello , how are you ?\n\nAssistant:"
mine_g = m.greedy(prompt).split()
recorded = None
for line in open("lens/runs/GPU-BEST.txt", encoding="utf-8", errors="replace"):
    if line.startswith("  GREEDY "):
        recorded = line.split("GREEDY", 1)[1].strip().split()
        break
if recorded is None:
    print("2) no GREEDY line found in lens/runs/GPU-BEST.txt (skipped)")
else:
    n = min(len(mine_g), len(recorded), 20)
    same = sum(1 for a, b in zip(mine_g[:n], recorded[:n]) if a == b)
    print("2) example greedy vs trainer GREEDY, first %d tokens: %d/%d identical" % (n, same, n))
    print("   example : %s" % " ".join(mine_g[:n]))
    print("   trainer : %s" % " ".join(recorded[:n]))
    if same != n:
        ok = False
        print("   FAIL: greedy decoding differs from the trainer's own output")

# ---- 3. wrong vocab must be refused, not silently mis-decoded
p = subprocess.run([sys.executable, "examples/inference_numpy.py", CKPT,
                    "data/chat/prepared/vocab_v12000.json", prompt],
                   capture_output=True, text=True)
refused = p.returncode != 0 and "does not belong to this checkpoint" in (p.stderr + p.stdout)
print("3) wrong vocab (12000 words) refused: %s (exit %d)" % (refused, p.returncode))
if not refused:
    ok = False
    print("   FAIL: a mismatched vocabulary was accepted")

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
