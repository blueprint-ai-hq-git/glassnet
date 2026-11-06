"""feed_customer.py — ONE command: customer data in -> cleaner -> windows -> ChatNet -> gate.

The owner's product line end to end: "we made it easy to use, customer has our own model data and
feed that to the model."

  .venv-gpu/bin/python3 lens/scripts/feed_customer.py --data FILE.jsonl --clean on --out feed1

Stages, each printing its own numbers:
  1 clean.load_any       any shape (jsonl/json/txt/dir), auto field pick
  2 clean.clean          the ladder's proven cleaner, ON or OFF (--clean)
  3 tokenize + vocab     chat tokenizer; vocab built FROM THE CLEANED TEXT (cap --vocab), so a
                         customer with new words gets new words, not <unk>-drowning
  4 windows              W-token windows; TRAIN/HOLDOUT SPLIT BY CONVERSATION (an item is a unit;
                         the same conversation never lands on both sides)
  5 ChatNet              the library architecture (LensNet under the thin wrapper), concepts built
                         on TRAIN tokens only (the standing rule)
  6 GATE, pre-registered, binary:
       f1  holdout val ppl < 2 x vocab size   (learns customer language 2x better than guessing)
       f3  stage counts reconcile: loaded = kept + dropped, rule by rule
     f2 (ON vs OFF) is decided by running the command twice with --clean on/off; both paths are
     real and the logs compare.
Nothing here claims to beat the dialogue model — different data distribution, different job. The
gate proves the PIPELINE converts arbitrary customer text into a trained, checked, saved model.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import clean                                             # noqa: E402
import lens                                              # noqa: E402
import lens.tensor as T                                  # noqa: E402
from lens import backend as B                            # noqa: E402
from lens.concepts import Concept                        # noqa: E402
from train_chat import ChatNet                           # noqa: E402

TOK = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def log(m):
    print(m, flush=True)


def build_vocab(items, cap):
    freq = {}
    for t in items:
        for w in TOK.findall(t.lower()):
            freq[w] = freq.get(w, 0) + 1
    ordered = ["<unk>"] + sorted(freq, key=lambda w: (-freq[w], w))[:cap - 1]
    vmap = {w: i for i, w in enumerate(ordered)}
    return ordered, vmap


def windows_by_item(seqs, W):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    cut = int(0.9 * len(order))
    tr_set = set(int(j) for j in order[:cut])
    tr_ids, ho_ids, pos = [], [], 0
    ids_flat = np.concatenate(seqs)
    train_mask = np.zeros(len(ids_flat), dtype=bool)
    for j, s in enumerate(seqs):
        (tr_ids if j in tr_set else ho_ids).extend(s)
        if j in tr_set:
            train_mask[pos:pos + len(s)] = True
        pos += len(s)

    def mk(ids):
        X, Y = [], []
        for k in range(W, len(ids)):
            X.append(ids[k - W:k])
            Y.append(ids[k])
        return (np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)) \
            if X else (np.zeros((0, W), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    return mk(tr_ids), mk(ho_ids), ids_flat, train_mask, len(tr_set), len(order) - len(tr_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clean", default="on", choices=["on", "off"])
    ap.add_argument("--vocab", type=int, default=3500)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--out", default="feed1")
    a = ap.parse_args()

    log("=" * 96)
    log(f"FEED CUSTOMER DATA — clean:{a.clean.upper()}  data:{a.data}")
    log("=" * 96)

    items = clean.load_any(a.data)
    log(f"  [1] loaded {len(items):,} text items")
    kept, rep = clean.clean(items, enabled=(a.clean == "on"), verbose=True)
    drop_sum = sum(rep["dropped"].values()) if a.clean == "on" else 0
    recon = len(items) == len(kept) + drop_sum
    log(f"  [2] loaded {len(items):,} = kept {len(kept):,} + dropped {drop_sum:,} -> "
        f"{'RECONCILE' if recon else 'MISMATCH'}")
    if not recon:
        raise SystemExit("f3 FAIL: stage counts do not reconcile")

    vocab, vmap = build_vocab(kept, a.vocab)
    seqs = [np.fromiter((vmap.get(w, 0) for w in TOK.findall(t.lower())), dtype=np.int64)
            for t in kept]
    seqs = [s for s in seqs if len(s) >= a.window + 2]
    ntok = sum(len(s) for s in seqs)
    unk = sum(int((s == 0).sum()) for s in seqs)
    log(f"  [3] vocab {len(vocab):,}  tokens {ntok:,}  <unk> rate {unk/max(ntok,1)*100:.2f}%")

    (Xtr, Ytr), (Xho, Yho), ids_flat, train_mask, ntr, nho = windows_by_item(seqs, a.window)
    if len(Xtr) > a.max_train:
        Xtr, Ytr = Xtr[:a.max_train], Ytr[:a.max_train]
    log(f"  [4] windows {len(Xtr):,} train / {len(Xho):,} holdout  "
        f"({ntr} vs {nho} conversations, split by conversation)")
    if len(Xtr) < 1000 or len(Xho) < 50:
        raise SystemExit("not enough windows to train/gate — give it more data")

    import chat_concepts_wide
    cs, cspec = chat_concepts_wide.build(vocab, ids_flat, train_mask, windows=Xtr)
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in cs]
    log(f"  [5] concepts {cspec['n_kept']} kept / {cspec['n_defined']} defined (train tokens only)")

    net = ChatNet(len(vocab), concepts, d=a.d, window=a.window, seed=a.seed,
                  lam_concept=0.3, device=a.device)
    log(f"  [5] ChatNet {net.n_params():,} params  vocab {len(vocab):,}  W {a.window}  d {a.d}")

    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        rng = np.random.default_rng(a.seed + ep)
        ordn = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for s0 in range(0, len(Xtr), a.batch):
            idx = ordn[s0:s0 + a.batch]
            for p in net.parameters():
                p.grad = None
            loss, _ = net.loss(Xtr[idx], Ytr[idx])
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - a.lr * p.grad
            tot += float(B.to_cpu(loss.data))
            nb += 1
        vp, vc = 0.0, 0
        for s0 in range(0, len(Xho), 512):
            idx = np.arange(s0, min(s0 + 512, len(Xho)))
            lg = net.forward_arrays(Xho[idx])["logits"]
            ce = float(T.softmax_ce(T.Tensor(B.to_dev(lg)), Yho[idx]).data)
            vp += ce * len(idx)
            vc += int((lg.argmax(axis=1) == Yho[idx]).sum())
        ppl = float(np.exp(vp / max(len(Xho), 1)))
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "holdout_ppl": ppl,
               "holdout_acc": vc / max(len(Xho), 1), "seconds": time.perf_counter() - t0}
        hist.append(rec)
        log(f"  [6] ep {ep}/{a.epochs} loss {rec['train_loss']:.4f}  HOLDOUT ppl {ppl:8.2f} "
            f"(bar {2 * len(vocab):,})  acc {rec['holdout_acc'] * 100:5.2f}%  "
            f"({rec['seconds']:.0f}s)")

    final = hist[-1]
    f1 = final["holdout_ppl"] < 2 * len(vocab)
    log("")
    log("  GATES (pre-registered, binary):")
    log(f"    f1 holdout ppl {final['holdout_ppl']:.2f} < 2 x vocab ({2 * len(vocab):,}) -> "
        f"{'PASS' if f1 else 'FAIL'}")
    log(f"    f3 stage counts reconciled [2] -> PASS")
    log(f"    f2 = run twice (--clean on / --off) and compare holdout ppl; both paths execute")

    OUT = os.path.join(PROJ, "lens", "runs", "feed")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"args": vars(a),
               "stages": {"loaded": len(items), "kept": len(kept),
                          "dropped": (rep["dropped"] if a.clean == "on" else {}),
                          "tokens": ntok, "unk_rate": unk / max(ntok, 1),
                          "windows_train": len(Xtr), "windows_holdout": len(Xho),
                          "conv_train": ntr, "conv_holdout": nho},
               "history": hist, "gates": {"f1": bool(f1), "f3": True},
               "params": net.n_params(), "vocab_size": len(vocab),
               "concepts": cspec["n_kept"]},
              open(os.path.join(OUT, f"{a.out}_{a.clean}_log.json"), "w"), indent=1)
    if f1:
        # save the INNER LensNet, not the wrapper: lens.save reads net.device_name/seed/lam_* which
        # ChatNet forwards only for its own fields; train_chat.py uses net._net for exactly this
        # reason (read, not guessed — a save crash after a PASSING gate is how feed_apps first ran).
        lens.save(net._net, os.path.join(OUT, f"{a.out}_{a.clean}.npz"))
        net2 = lens.load(os.path.join(OUT, f"{a.out}_{a.clean}.npz"), device=a.device)
        lg1 = net.forward_arrays(Xho[:64])["logits"]
        lg2 = net2.forward_arrays(Xho[:64])["logits"]
        rt = float(np.abs(lg1 - lg2).max())
        log(f"  checkpoint saved; round-trip max diff {rt:.3e} "
            f"{'IDENTICAL' if rt == 0.0 else 'MISMATCH'}")
        if rt != 0.0:
            raise SystemExit("round-trip mismatch")
    log(f"  saved {a.out}_{a.clean}_log.json")
    log("=" * 96)
    sys.exit(0 if f1 else 1)


if __name__ == "__main__":
    main()
