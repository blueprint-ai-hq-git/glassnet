"""structural check: does LensNet(attention=True, direct=True, tied) build, forward, and keep the two-part
readout decomposition EXACT? CPU only, seconds, no training, no file written except its own stdout.

WHY (2026-10-22 21:2x self-wake): P65c measured that the best ship "loses the thread after ONE token of its own
output" (free-running 7.8 -> 2.0 -> 1.2 ... % against teacher-forced 7.8/8.6/13.7/...). Both best checkpoints
have attention=False, i.e. the readout sees a POSITION-WEIGHTED sum of the window embeddings with no
content-based routing at all. Enabling attention is the natural next architectural lever. Before that arm can be
queued it must be shown that the combination attention=True + direct_mode='tied' (a) constructs, (b) forwards
without error, (c) has finite logits, and (d) keeps the readout split exact - AND that the split's direct term
must then use the ATTENDED window (out['window_input']), not emb[ctx], because attention transforms x. That last
point is a correction to the P75/P77 instruments' assumption, which was valid only because attention=False.
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

from lens import backend as BE                                  # noqa: E402
from lens.concepts import Concept                               # noqa: E402
from lens.model import LensNet                                  # noqa: E402

V, D, W = 12002, 64, 32          # tiny d on purpose: this checks CONSTRUCTION/forward, not quality
concepts = [Concept("is_a_question", [18, 25, 40]), Concept("mentions_the_user", [7, 12])]

for attn in (False, True):
    net = LensNet(V, concepts, d=D, window=W, seed=0, lam_concept=0.3, lam_act=0.0,
                  attention=attn, direct=True, direct_mode="tied")
    net.device_name = "cpu"
    n_par = sum(int(np.asarray(p.data).size) for p in net.parameters())
    print(f"attention={attn}  direct=tied: params={n_par:,}  "
          f"attn={'present' if net.attn is not None else 'None'}  dmix={None if net.dmix is None else tuple(net.dmix.data.shape)}")
    rng = np.random.default_rng(0)
    ctx = np.ascontiguousarray(rng.integers(1, V, size=(1, W)).astype(np.int64))
    out = net.forward(ctx)
    lg = np.asarray(BE.to_cpu(out["logits"].data), dtype=np.float64)
    h = np.asarray(BE.to_cpu(out["concepts"].data), dtype=np.float64)
    x = np.asarray(BE.to_cpu(out["window_input"].data), dtype=np.float64)      # THE ATTENDED WINDOW when attn
    e = np.asarray(net.emb.data, dtype=np.float64)
    aW = np.asarray(net.answer.W.data, dtype=np.float64)
    ab = np.asarray(net.answer.b.data, dtype=np.float64)
    dm = np.asarray(net.dmix.data, dtype=np.float64)
    zc = h @ aW + ab
    zd_attended = (np.maximum(x.reshape(1, -1) @ dm, 0.0) @ e.T)
    zd_rawemb = (np.maximum(e[np.asarray(ctx)].reshape(1, -1) @ dm, 0.0) @ e.T)
    r_att = float(np.max(np.abs(zc + zd_attended - lg)))
    r_raw = float(np.max(np.abs(zc + zd_rawemb - lg)))
    print(f"   logits finite: {bool(np.isfinite(lg).all())}   shape {lg.shape}   "
          f"residual using window_input {r_att:.3e}   residual using emb[ctx] {r_raw:.3e}")
    print(f"   -> split {'EXACT with window_input' if r_att < 1e-4 else 'NOT exact with window_input'}; "
          f"{'emb[ctx] would be WRONG' if r_raw > 1e-4 else 'emb[ctx] coincides (attention off)'}")
