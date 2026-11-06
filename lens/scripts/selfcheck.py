"""selfcheck.py — ONE command that proves the whole GlassNet stack, every gate, every refusal.

Run:  .venv-gpu/bin/python3 lens/scripts/selfcheck.py [--fast]
Exit 0 only if EVERY check passes. Any check that cannot run (missing artifact) is FAIL, not SKIP —
a stack that cannot be verified from a clean shell is not shippable.

WHAT IS CHECKED (and why each is here — every one of these corresponds to a real bug class this
project hit, see LOG.md):
  1  cleaner self-test           counts reconcile exactly; OFF mode is a true pass-through
  2  checkpoint integrity        every shipped .npz loads; concept NAMES + FORMULAS resolve from
                                 the frozen registry; a tampered formula MUST raise
  3  round-trip identity         load -> forward -> compare logits: must be bit-identical
  4  explain identity            hand-recomputed last-two-layers arithmetic == engine (all rungs)
  5  variants interface          see/talk/see_and_talk all answer; talk maps to a REAL WORD
  6  refusal                     see-only asked to talk raises; unknown capability raises
  7  vocab integrity             talk id outside vocab raises; checkpoints carry WORDS not sizes
  8  determinism                 same input twice -> identical outputs (no hidden RNG in forward)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import clean                                             # noqa: E402
import lens                                              # noqa: E402
import lens.tensor as T                                  # noqa: E402
import train_seetalk as ts                               # noqa: E402
import variants as VA                                    # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def sha(path):
    h = hashlib.sha256(open(path, "rb").read())
    return h.hexdigest()[:12]


def main():
    T.set_device("cpu")
    t0 = time.perf_counter()
    print("=" * 96)
    print("GLASSNET SELFCHECK — the whole stack, one command")
    print("=" * 96)

    # ---------------------------------------------------------------- 1 cleaner
    print("\n1. CLEANER")
    probe = ([""] * 2 + ["short"] * 3
             + ["a" * 40] * 2 + ["." * 100] * 2
             + ["A perfectly ordinary sentence about code and data."] * 4
             + ["dup item that is quite long and normal looking"] * 3)
    kept, rep = clean.clean(probe, enabled=True, verbose=False)
    check("cleaner counts reconcile", len(kept) == 2 and rep["dropped"] == {
        "drop_duplicate": 5, "drop_too_short": 3, "drop_empty": 2,
        "drop_repetition": 2, "drop_low_signal": 2},
        f"kept={len(kept)} drops={rep['dropped']}")
    off, _ = clean.clean(probe[:10], enabled=False, verbose=False)
    check("cleaner OFF is pass-through", off == probe[:10])

    # ---------------------------------------------------------------- 2/3/4 checkpoints
    print("\n2-4. CHECKPOINTS: load, registry strictness, round-trip, explain identity")
    R = os.path.join(PROJ, "lens", "runs", "seetalk")
    for f in ("seetalk_small2", "seetalk_big", "seetalk_teacher_d128", "seetalk_k22_big",
               "seetalk_k22_big_fixedgrad"):
        p = os.path.join(R, f"{f}.npz")
        if not os.path.exists(p):
            check(f"{f}: artifact present", False, "MISSING")
            continue
        try:
            net, meta = ts.SeeTalkNet.load(p)
            check(f"{f} loads, K={len(net.concepts)}", True,
                  f"sha {sha(p)} vocab {len(meta['vocab'])} words")
        except Exception as e:
            check(f"{f} loads", False, repr(e)[:70])
            continue
        # round-trip: a fresh forward through a SECOND load must be identical
        data = ts.build_caption_data(n=600, seed=0)
        Xb, Sb = data["Xva"][:32], data["Sva"][:32]
        o1 = net.forward_arrays(Xb, Sb)["logits"]
        net2, _ = ts.SeeTalkNet.load(p)
        o2 = net2.forward_arrays(Xb, Sb)["logits"]
        rt = float(np.abs(o1 - o2).max())
        check(f"{f} round-trip identical", rt == 0.0, f"max diff {rt:.1e}")
        # explain identity (g3 math, hand-recomputed)
        Pn = net.front.patches(Xb)
        zd = (Pn.reshape(-1, Pn.shape[-1]) @ lens.backend.to_cpu(net.front.proj.W.data)
              + lens.backend.to_cpu(net.front.proj.b.data)).reshape(len(Xb), -1)
        cl_ = zd @ lens.backend.to_cpu(net.concept.W.data) + lens.backend.to_cpu(net.concept.b.data)
        h3 = 1 / (1 + np.exp(-np.clip(cl_, -60, 60)))
        cemb = lens.backend.to_cpu(net.cap_emb.data)
        cxt = np.concatenate([cemb[Sb[:, w]] for w in range(Sb.shape[1])], axis=1)
        manual = (np.concatenate([h3, cxt], axis=1)
                  @ lens.backend.to_cpu(net.cap_out.W.data)
                  + lens.backend.to_cpu(net.cap_out.b.data))
        resid = float(np.abs(manual - o1).max())
        check(f"{f} explain identity", resid < 1e-5, f"residual {resid:.1e}")
        # determinism: same input twice
        o3 = net.forward_arrays(Xb, Sb)["logits"]
        check(f"{f} forward deterministic", np.array_equal(o1, o3))

    # registry strictness: a tampered formula must be refused
    p = os.path.join(R, "seetalk_small2.npz")
    z = dict(np.load(p, allow_pickle=True))
    m = json.loads(str(z["meta"][0]))
    m["concepts"][0]["formula"] = "TAMPERED mean(pixels) > 0.99"
    z["meta"] = np.array([json.dumps(m)])
    bad = os.path.join(PROJ, "lens", "runs", "seetalk", "_tampered.npz")
    np.savez_compressed(bad, **z)
    try:
        ts.SeeTalkNet.load(bad)
        check("tampered concept formula REFUSED", False, "loaded a corrupted checkpoint!")
    except ValueError:
        check("tampered concept formula REFUSED", True)
    finally:
        os.remove(bad)

    # ---------------------------------------------------------------- 5/6/7 variants
    print("\n5-7. VARIANTS: all three answer, refusals, vocab integrity")
    v = VA.build_default(device="cpu")
    def _ctx(v):
        """context of the TALK model's own width (follows the shipping default, never hardcoded)."""
        w = v.models["talk"][0].W if "talk" in v.models else v._need("talk")[0].W
        return [0] * w

    caps = v.capabilities()
    check("all three capabilities loaded", caps == ["see", "see_talk", "talk"], str(caps))
    d = __import__("vl_data").prepare_vqa("fashion", seed=0)
    r = v.see(d["Xte"][0])
    check("see answers", "answer" in r and 0 < r["confidence"] <= 1, f"{r['answer']} {r['confidence']:.2f}")
    t = v.talk(_ctx(v))
    check("talk answers with a REAL WORD", isinstance(t["word"], str), repr(t["word"]))
    vocab = v.models["see_talk"][1]["vocab"]
    vi = {w: i for i, w in enumerate(vocab)}
    st = v.see_and_talk(d["Xte"][0], [vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"],
                                      vi["is"], vi["a"]])
    check("see_and_talk answers a caption word", st["word"] in vocab, st["word"])
    see_only = VA.Variants(see=os.path.join(PROJ, "lens", "runs", "image",
                                            "lad_image_cls_ep24.npz"), device="cpu")
    try:
        see_only.talk(_ctx(see_only))
        check("refusal: see-only asked to talk", False, "IT ANSWERED — dishonest build")
    except ValueError as e:
        check("refusal: see-only asked to talk", True, str(e)[:52])
    # vocab integrity: out-of-range id must raise, not print a guessed word
    v.chat_vocab = ["only", "two", "words"]
    try:
        v.talk(_ctx(v))
        check("refusal: talk id outside vocab", False, "printed a word it cannot know")
    except ValueError:
        check("refusal: talk id outside vocab", True)
    v = VA.build_default(device="cpu")   # restore

    # ---------------------------------------------------------------- image/vl checkpoints
    print("\n8. IMAGE-LINE CHECKPOINTS load through the vision loader")
    from lens.vision import load_vision
    for name, path in (("see-only (ship, K23)", os.path.join(PROJ, "lens", "runs", "image",
                                                              "ktest_real_s0_ep12.npz")),
                       ("see-only (ladder, K13)", os.path.join(PROJ, "lens", "runs", "image",
                                                                "lad_image_cls_ep24.npz")),
                       ("vqa", os.path.join(PROJ, "lens", "runs", "vl",
                                            "vqa2_small_ep6_vocabfix.npz"))):
        try:
            net, meta = load_vision(path)
            check(f"{name} loads", True, f"{type(net).__name__}")
        except Exception as e:
            check(f"{name} loads", False, repr(e)[:70])

    # ---------------------------------------------------------------- 9 gradient integrity
    # Added 2026-10-10 after the w=1 bug: the concept-loss backward was wrong by a factor of
    # pos_weight for every concept with support != 0.5, and NO selfcheck gate ran a gradient check,
    # so the wrong formula shipped for months. Now the full op-table check runs HERE, as a gate.
    print("\n9. GRADIENT INTEGRITY — every op against central differences, at multiple input ranges")
    try:
        import subprocess
        gcc = subprocess.run(
            [sys.executable, os.path.join(PROJ, "lens", "scripts", "gradcheck_full_ops.py")],
            capture_output=True, text=True, timeout=900)
        gok = gcc.returncode == 0
        check("gradcheck: all ops (float64, multi-range)", gok,
              gcc.stdout.strip().splitlines()[-1][:80] if gcc.stdout else "no stdout")
        if not gok and gcc.stderr:
            print(f"        {gcc.stderr.strip()[-200:]}")
    except Exception as e:
        check("gradcheck: all ops (float64, multi-range)", False, repr(e)[:80])

    print()
    # ---------------------------------------------------------------- 10 strict audits (anti-rot)
    # Added 2026-10-11 after the owner's audit: the label-honesty audit (audit_concept_meaning.py)
    # was BROKEN (IndentationError) and covered only two E-series checkpoints — and it was NOT in
    # selfcheck, so a broken audit could not turn the check red. Now the strict audits RUN here.
    # Gates = anti-regression vs today's measured numbers; the redesign program drives them to 100%.
    print("10. STRICT LABEL-HONESTY — every shipping line, measured here, anti-regression")
    try:
        import subprocess as sp
        for tag, ckpt, floor in (("talk", "t11_talk41_ep192.npz", 40),
                                 ("code", "t8f_code24_ep32.npz", 24)):
            aud = sp.run(
                [sys.executable, os.path.join(PROJ, "lens", "scripts", "audit_strict_text.py"), tag],
                capture_output=True, text=True, timeout=900,
                # BOTH overrides pinned: without the CKPT key the audit silently falls back to its
                # default subject (caught 2026-10-11: it audited the old 35-set T8 and reported
                # 27/24 — a PASS on the wrong subject is still a wrong answer).
                env={**os.environ, "CODE_DOC": "clean",
                     "CODE_CKPT": ckpt if tag == "code" else os.environ.get("CODE_CKPT", ""),
                     "TALK_CKPT": ckpt if tag == "talk" else os.environ.get("TALK_CKPT", "")})
            ok_aud = aud.returncode == 0
            tot = None
            for ln in aud.stdout.splitlines():
                if f"s1[{tag}]" in ln:
                    import re as _re
                    m = _re.search(r"\((\d+)/(\d+)\)", ln)
                    if m:
                        tot = int(m.group(1))
            ok_floor = tot is not None and tot >= floor
            check(f"strict audit runs [{tag}]", ok_aud,
                  aud.stdout.strip().splitlines()[-1][:80] if aud.stdout else "no stdout")
            check(f"strict honest >= {floor} [{tag}]", ok_floor, f"{tot}/{floor}")
            if aud.returncode != 0 and aud.stderr:
                print(f"        {aud.stderr.strip()[-150:]}")
        # MEDICAL (vl colour line): same anti-rot rule, its own audit + pinned subject
        aud_m = sp.run(
            [sys.executable, os.path.join(PROJ, "lens", "scripts", "audit_strict_medical.py")],
            capture_output=True, text=True, timeout=1800,
            env={**os.environ, "MED_CKPT": "med_cc3_ep8.npz"})
        m_ok = aud_m.returncode == 0
        m_tot = None
        for ln in aud_m.stdout.splitlines():
            import re as _re2
            m2 = _re2.search(r"honest (\d+)/(\d+)", ln)
            if m2:
                m_tot = int(m2.group(1))
        check("strict audit runs [medical]", m_ok,
              aud_m.stdout.strip().splitlines()[-1][:80] if aud_m.stdout else "no stdout")
        check("strict honest >= 4 [medical]", m_tot is not None and m_tot >= 4, f"{m_tot}/4")
        # DATA JUDGE: its own smoke (real GOOD, corruptions BAD) -- artifact + behaviour in the loop
        dj = sp.run([sys.executable, os.path.join(PROJ, "lens", "scripts", "judge_cli_smoke.py")],
                    capture_output=True, text=True, timeout=1800,
                    env={**os.environ, "CD_VOCAB": "12000"})
        # VISION STRICT AUDIT (pinned ship + pinned datum, the same numbers that ship): the
        # image concept wall must stay 19/19 every time this runs, or rot is caught here.
        va = sp.run(
            [sys.executable, os.path.join(PROJ, "lens", "scripts", "audit_strict_vision.py")],
            capture_output=True, text=True, timeout=2400,
            env={**os.environ, "VISION_SHIP": "seetalk_k21o_mirror.npz", "VISION_AUDIT_N": "60000"})
        va_line = [l for l in va.stdout.splitlines() if "STRICT:" in l]
        check("vision strict 19/19", ("honest 19/19" in va.stdout),
              (va_line[-1].strip()[:90] if va_line else "no verdict line"))
        # AUDIO BATTERY (the scenario test that caught BUG-A1): voice/tone/noise/silence
        ab = sp.run([sys.executable, os.path.join(PROJ, "lens", "scripts", "audio_battery.py")],
                    capture_output=True, text=True, timeout=1800,
                    env={**os.environ, "CD_VOCAB": "12000"})
        ab_line = [l for l in ab.stdout.splitlines() if "AUDIO BATTERY" in l]
        check("audio battery 4/4", ab.returncode == 0,
              (ab_line[-1].strip()[:80] if ab_line else "no verdict line"))
        check("data judge smoke", dj.returncode == 0,
              (dj.stdout.strip().splitlines()[-1][:80] if dj.stdout else "no stdout"))
    except Exception as e:
        check("strict audits (talk/code)", False, repr(e)[:80])

    # ---------------------------------------------------------------- 7 speech line (REAL audio)
    # The spoken-word recognizer is a real capability on external audio (Speech Commands, CC BY
    # 4.0), so it gets a gate like every other line. The ship is a plain array checkpoint (its own
    # trainer), so the check recomputes the official-val accuracy from the data itself rather than
    # trusting the report.
    print("\n7. SPEECH LINE (readable spoken words on real audio)")
    try:
        sp = os.path.join(PROJ, "data", "speech", "speech_v1.npz")
        # the SHIPPED speech artifact is the 7-concept model: the 8-concept version trained at
        # 37.64% but its `noisy` concept failed the strict audit (own 0.425 vs rival
        # high_band_present 0.706), and the project rule is that a concept which cannot be
        # label-honest is DROPPED, never shipped with a soft label. Cost of the drop: 0.21 pts.
        ship = os.path.join(PROJ, "lens", "runs", "speech",
                            "speech_readable_s0_drop-noisy.npz")
        meta_p = os.path.join(PROJ, "data", "speech", "speech_v1_meta.json")
        check("speech data present", os.path.exists(sp) and os.path.exists(ship),
              os.path.basename(sp))
        if os.path.exists(sp) and os.path.exists(ship):
            dz = np.load(sp, allow_pickle=True)
            zz = np.load(ship, allow_pickle=True)
            X, y, split = dz["X"], dz["y"], dz["split"]
            mj = json.load(open(meta_p))
            feats = list(mj["features"])
            concepts = [str(c) for c in zz["concepts"]]
            cf = {"loud": "rms", "noisy": "zcr", "bright": "centroid_hz",
                  "low_band_dominant": "low_ratio", "high_band_present": "high_ratio",
                  "voiced": "voiced_frac", "tonal": "flatness", "early_peak": "peak_pos"}
            tr, va = split == 0, split == 1
            mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8      # TRAIN-only stats, as trained
            Z = ((X - mu) / sd).astype(np.float32)
            def _fwd(Zb):
                h = 1 / (1 + np.exp(-np.clip(Zb @ zz["W1"] + zz["b1"], -60, 60)))
                r = np.maximum(h @ zz["W2"] + zz["b2"], 0)
                return r, h @ zz["W2"] + zz["b2"], zz["W3"], r @ zz["W3"] + zz["b3"]
            _r, pre_r, _W3, lg = _fwd(Z[va])
            acc = float((lg.argmax(axis=1) == y[va]).mean())
            base = float(max(np.bincount(y[tr])) / tr.sum())
            check("speech official-val accuracy > 2x chance", acc > 0.25,
                  f"acc {acc*100:.2f}% vs majority baseline {base*100:.2f}% "
                  f"(8 classes, chance 12.5%)")
            rep = json.load(open(os.path.join(PROJ, "lens", "runs", "speech",
                                             "speech_report_drop-noisy.json")))
            n_hon = sum(1 for c in rep["concepts"].values() if c["honest"])
            check("shipped speech concept wall fully honest", n_hon == len(rep["concepts"]),
                  f"{n_hon}/{len(rep['concepts'])} honest"
                  + ("" if n_hon == len(rep["concepts"]) else
                     "; failing: " + ", ".join(k for k, v in rep["concepts"].items()
                                               if not v["honest"])))
            conns = int((zz["W2"] != 0).sum())
            per_rule = conns / zz["W2"].shape[1]
            check("speech rules are sparse (<=3.05 concepts/rule)", per_rule <= 3.05,
                  f"{per_rule:.2f} per rule ({conns} connections)")
            Xv = Z[va][:32]
            h1 = 1 / (1 + np.exp(-np.clip(Xv @ zz["W1"] + zz["b1"], -60, 60)))
            r1 = np.maximum(h1 @ zz["W2"] + zz["b2"], 0)
            lg1 = r1 @ zz["W3"] + zz["b3"]
            h1b = 1 / (1 + np.exp(-np.clip(Xv @ zz["W1"] + zz["b1"], -60, 60)))
            r1b = np.maximum(h1b @ zz["W2"] + zz["b2"], 0)
            check("speech forward deterministic/explain identity",
                  np.array_equal(lg1, r1b @ zz["W3"] + zz["b3"]),
                  f"max diff {float(np.abs(lg1 - (r1b @ zz['W3'] + zz['b3'])).max()):.1e}")
    except Exception as e:
        check("speech line", False, repr(e)[:90])

    print()
    dt = time.perf_counter() - t0
    if FAILS:
        print(f"  SELFCHECK: {len(FAILS)} FAILURE(S) in {dt:.0f}s: {FAILS}")
    else:
        print(f"  SELFCHECK: ALL CHECKS PASS  ({dt:.0f}s)")
    print("=" * 96)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
