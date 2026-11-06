import sys, numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import lens.api as api
from lens.concepts import Concept
vocab, window, d, K = 50, 8, 16, 8
rng = np.random.default_rng(0)
X = rng.integers(0, vocab, size=(3000, window)); y = X[:, window//2]
Xv = rng.integers(0, vocab, size=(1000, window)); yv = Xv[:, window//2]
cs = [Concept(f"cat{k}", [t for t in range(vocab) if t % K == k]) for k in range(K)]
for mode in ("lowrank", "tied"):
    net = api.Net(vocab, cs, d=d, window=window, seed=1, device="cpu", lr=0.5, direct=True, direct_mode=mode)
    npar = sum(int(np.prod(p.data.shape)) for p in net.parameters())
    api.train(net, X, y, epochs=8, batch=64, lr=0.5, seed=1)
    acc = float((np.asarray(net.forward(Xv)["logits"].data).argmax(axis=1) == yv).mean())
    print(f"  direct_mode={mode:8s} identity task held-out {acc*100:5.1f}%  params {npar:,}")
