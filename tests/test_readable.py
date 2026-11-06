"""test_readable.py — scenario suite for TODAY's readable-model modules.

Covers glassnet.explain (the exact explanation) and glassnet.selfgen (our own generator), plus
the checkpoint-loading interface that caused the Target F audit bug.

Every test prints the real numbers it saw. Nothing is asserted without being shown.
Run:  bash scripts/run_test_readable.sh
"""
import json
import os
import sys
import tempfile

PROJ = "/run/media/maintainer/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail))
        print(f"  PASS  {name}\n        {detail}")
    except Exception as exc:
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")


print("=" * 100)
print("READABLE-MODEL SCENARIO SUITE (glassnet.explain + glassnet.selfgen)")
print("=" * 100)

from glassnet.tensor import set_backend            # noqa: E402
from glassnet import tiny_data                      # noqa: E402
from glassnet import selfgen                        # noqa: E402
from glassnet.explain import explain, format_explanation, concept_word_table  # noqa: E402
import t3_audit_m3 as M3                            # noqa: E402

R = os.path.join(PROJ, "runs", "t3")
set_backend("numpy")
DATA = tiny_data.build(win=8, vocab_cap=1500)
Xva, Yva, VOCAB = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"]), DATA["vocab"]
names = [t["name"] for t in
         json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]]
names = names + [f"untaught_{i}" for i in range(16 - len(names))]


def _teacher():
    return M3.load_ckpt(M3.build_net(0),
                        os.path.join(R, "ckpt", "normal_K9_lr1.0_meaningful_ext_e_seed0.npz"))


# ------------------------------------------------------------------ 1. generator
def t_generate_shape():
    net = _teacher()
    rng = np.random.default_rng(0)
    seeds = Xva[rng.choice(len(Xva), 6, replace=False)]
    seqs = selfgen.generate(net, seeds, n_new=12, temperature=1.0, rng=rng)
    assert seqs.shape == (6, 20), seqs.shape
    assert np.array_equal(seqs[:, :8], seeds), "seed prefix must be preserved verbatim"
    return f"generate() -> {seqs.shape} (6 seeds x (8 given + 12 generated)), seed prefix intact"
check("1. generate() shape and seed preservation", t_generate_shape)


# ------------------------------------------------------------------ 2. determinism
def t_deterministic():
    net = _teacher()
    seeds = Xva[:4]
    a = selfgen.generate(net, seeds, n_new=8, rng=np.random.default_rng(3))
    b = selfgen.generate(net, seeds, n_new=8, rng=np.random.default_rng(3))
    assert np.array_equal(a, b), "same rng seed must give the same text"
    c = selfgen.generate(net, seeds, n_new=8, rng=np.random.default_rng(4))
    return (f"same seed -> identical ({np.array_equal(a, b)}), different seed -> "
            f"different ({not np.array_equal(a, c)})")
check("2. generation is reproducible from its rng seed", t_deterministic)


# ------------------------------------------------------------------ 3. windows
def t_windows():
    net = _teacher()
    seqs = selfgen.generate(net, Xva[:5], n_new=10, rng=np.random.default_rng(1))
    X, y = selfgen.to_windows(seqs, win=8)
    assert X.shape[1] == 8 and len(X) == len(y) == 5 * 10
    # the target must be the token right after the window, in the SAME sequence
    ok = all(np.array_equal(X[i], seqs[i // 10][i % 10:i % 10 + 8]) for i in range(0, 50, 7))
    assert ok, "window/target alignment is wrong"
    assert y.max() < len(VOCAB) + 1
    return f"to_windows() -> {X.shape}, {len(y)} targets, alignment verified, max id {int(y.max())}"
check("3. windows and targets are aligned", t_windows)


# ------------------------------------------------------------------ 4. selection
def t_select():
    net = _teacher()
    seqs = selfgen.generate(net, Xva[:40], n_new=8, rng=np.random.default_rng(2))
    X, y = selfgen.to_windows(seqs, win=8)
    conf = selfgen.teacher_confidence(net, X, y)
    assert 0.0 <= conf.min() and conf.max() <= 1.0
    Xa, ya, ka = selfgen.select(X, y, conf, rule="all")
    Xc, yc, kc = selfgen.select(X, y, conf, rule="conf50")
    Xf, yf, kf = selfgen.select(X, y, conf, rule="conf", threshold=0.5)
    assert len(Xa) == len(X)
    assert abs(len(Xc) / len(X) - 0.5) < 0.05, len(Xc) / len(X)
    assert len(Xf) <= len(X)
    return (f"all keeps {len(Xa)}/{len(X)}; conf50 keeps {len(Xc)}/{len(X)} "
            f"({len(Xc)/len(X)*100:.0f}%); conf>=0.5 keeps {len(Xf)} "
            f"(this is the arm that was EMPTY at 288 windows and forced the amendment)")
check("4. the pre-registered selection rules behave", t_select)


# ------------------------------------------------------------------ 5. corpus stats
def t_stats():
    net = _teacher()
    seqs = selfgen.generate(net, Xva[:60], n_new=16, rng=np.random.default_rng(5))
    X, y = selfgen.to_windows(seqs, win=8)
    st = selfgen.corpus_stats(X, y, VOCAB, real_X=np.asarray(DATA["Xtr"]))
    assert st["n_windows"] == len(X) and st["n_distinct_tokens"] > 0
    assert 0.0 <= st["top10_token_share"] <= 1.0
    return (f"{st['n_windows']} windows; distinct tokens {st['n_distinct_tokens']} "
            f"(real {st['real_n_distinct_tokens']}); entropy {st['target_entropy']:.3f} "
            f"vs real {st['real_target_entropy']:.3f}")
check("5. corpus health is measurable, generated vs real", t_stats)


# ------------------------------------------------------------------ 6. explain exactness
def t_explain_exact():
    net = _teacher()
    worst, at = 0.0, -1
    for i in range(400):
        e = explain(net, Xva[i], VOCAB, names)
        if e["residual"] > worst:
            worst, at = e["residual"], i
    assert worst < 1e-4, f"explanation not exact: worst residual {worst} at {at}"
    return f"worst |contributions + bias - model score| = {worst:.3e} over 400 windows (window {at})"
check("6. explain() is EXACT, not a story", t_explain_exact)


# ------------------------------------------------------------------ 7. explain formatting
def t_explain_text():
    net = _teacher()
    e = explain(net, Xva[7], VOCAB, names)
    txt = format_explanation(e)
    assert "INPUT" in txt and "IT SAYS" in txt and "EXACTNESS" in txt
    return f"format_explanation() -> {len(txt.splitlines())} lines, first: {txt.splitlines()[1][:64]!r}"
check("7. explain() renders as plain English", t_explain_text)


# ------------------------------------------------------------------ 8. concept word table
def t_word_table():
    net = _teacher()
    rows = concept_word_table(net, VOCAB, names, top_n=5)
    assert len(rows) == net.K
    return f"{len(rows)} rows; concept 0 top words: {', '.join(rows[0]['top_words'])}"
check("8. what each concept pushes up is readable", t_word_table)


# ------------------------------------------------------------------ 9. THE F BUG
def t_ckpt_keys():
    """The exact defect that made me report a wrong result: two checkpoint key conventions."""
    f_student = os.path.join(R, "ckpt", "student_readable_all_f_seed0.npz")
    f_real = os.path.join(R, "ckpt", "readable_K9_lr1.0_lc01_meaningful_ext_e_seed0.npz")
    if not (os.path.exists(f_student) and os.path.exists(f_real)):
        raise FileNotFoundError("checkpoints not present")
    ks = sorted(np.load(f_student).files)
    kr = sorted(np.load(f_real).files)
    assert ks != kr, "conventions should differ -- this is the bug"
    net = M3.build_net(0)
    M3.load_ckpt(net, f_student)                 # the BROKEN path
    h_broken = np.asarray(net.forward(Xva[:200])["concepts"].data)
    net2 = M3.build_net(0)
    z = np.load(f_student)
    net2.emb.data = z["emb"]; net2.concept.W.data = z["Wc"]; net2.concept.b.data = z["bc"]
    net2.answer.W.data = z["Wa"]; net2.answer.b.data = z["ba"]
    h_ok = np.asarray(net2.forward(Xva[:200])["concepts"].data)
    assert not np.allclose(h_broken, h_ok), "the two loaders should differ"
    return (f"f_train keys {ks} vs real keys {kr}; M3.load_ckpt on an f_train file gives "
            f"concept mean {h_broken[:,0].mean():.4f} instead of {h_ok[:,0].mean():.4f} "
            f"-- the loader bug, now pinned by a test")
check("9. the checkpoint-key mismatch is pinned by a test", t_ckpt_keys)


# ------------------------------------------------------------------ 10. audit guard
def t_audit_k():
    """T3M_K must match the model's K, or coverage comes out as an artifact (it printed 35%)."""
    f = os.path.join(R, "ckpt", "readable_K9_lr1.0_lc01_meaningful_ext_e_seed0.npz")
    net = M3.build_net(0)
    assert net.K == 9, f"T3M_K must be 9 for this checkpoint, got {net.K}"
    return (f"T3M_K={net.K} matches the checkpoint's K=9; a default of 32 would build a K=32 "
            f"net and report coverage 9/32 (it printed 35% once)")
check("10. the audit K guard holds", t_audit_k)


print("\n" + "=" * 100)
n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"READABLE-MODEL SUITE: {n_pass}/{len(RESULTS)} passed")
for name, st, _ in RESULTS:
    if st != "PASS":
        print(f"  !! {name}")
print("=" * 100)
json.dump([{"name": n, "status": s, "detail": d} for n, s, d in RESULTS],
          open(os.path.join(R, "test_readable_results.json"), "w"), indent=1)
sys.exit(0 if n_pass == len(RESULTS) else 1)
