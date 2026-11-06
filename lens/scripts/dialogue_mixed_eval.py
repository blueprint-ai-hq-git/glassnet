"""dialogue_mixed_eval.py — WHAT DOES A STORED CHECKPOINT SCORE ON EACH POPULATION OF ITS HELD-OUT SET?

PREDICTIONS P108a-h WERE WRITTEN BEFORE THIS FILE EXISTED:
    lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt

WHAT IT MEASURES, AND WHY (from files, not from taste)
  The corpus the project trains and validates on is a MERGE [data/chat/prepared_mix_ctrl_honest/meta_v12000.json]:
  data/chat/prepared_ext_honest (156,520 conversations: hh-rlhf, oasst1, ultrachat, orca-math — human dialogue)
  followed by data/chat/prepared_kn2 (584,145 conversations: metamath, cosmopedia textbooks/math, hendrycks,
  OpenR1-Math, gutenberg, magicoder, codefeedback, opencode — maths/textbook/code text). P105a
  [lens/runs/CORPUS-MIX-AUDIT.txt] verified the two groups are contiguous in that order, and counted that
  168,754,291 of the 197,204,306 assistant-turn tokens the loss trains on = 85.57% are kn2.
  The held-out column every arm quotes ("held-out 35.26% / 35.72% / 36.69% by position, n=58,191") is a number
  over BOTH groups at once, while the ship criterion (a reply to a chat prompt, lens/runs/PADFREE-TALK-*.txt)
  is entirely dialogue. This instrument prints the split for stored checkpoints.

EXACTLY WHAT IT DOES (and what it deliberately does NOT do)
  * val windows: the trainer's own call, chat_data.make_windows(ids, bounds, split, W, which=1,
    return_index=True) — same stride, same no-crossing-boundaries rule, so the window set is the trainer's.
  * assistant mask BY POSITION, the trainer's own rule (train_chat.py ~line 600/1264): a window target counts
    iff the token at that target's own corpus position is inside an assistant turn. The turn walk uses the same
    ChatML (`<|im_start|>` role / `<|im_end|>`) and legacy (`assistant :` / `human :`) markers. `--train-turn-end`
    is a TRAINING weight and is deliberately NOT added here, exactly as in the trainer's val column.
    DECLARED DEVIATION (checked by P108a): the walk is per val conversation with in_asst initialised False at
    the start of each conversation, where the trainer carries the flag across conversation boundaries.
  * group: conversation index < 156,520 = ext_honest = HUMAN DIALOGUE; else kn2. The boundary is READ from
    meta_v12000.json's own per_source_corpus counts, never hardcoded.
  * NO generation, NO training, NO edits: inputs are read-only, output is printed to stdout (the runner appends
    it to lens/runs/DIALOGUE-VS-MIXED.txt).

USAGE
    CKPT_LIST="label=rel/or/abs/path;label2=path2" EVAL_DEVICE=cpu bash lens/scripts/run_dialogue_mixed_eval.sh
  ENV: CHAT_DATA_DIR (default data/chat/prepared_mix_ctrl_honest), EVAL_DEVICE (cpu|gpu), WIN (default 32),
       EVAL_BATCH (default 512), DM_SCOPE (a free-text label printed in the header).
"""
import json
import os
import sys
import time

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

import chat_data                                              # noqa: E402
from lens import backend as B                                 # noqa: E402
from lens import tensor as T                                  # noqa: E402
from lens.concepts import Concept                             # noqa: E402
from lens.model import LensNet                                # noqa: E402

W = int(os.environ.get("WIN", "32"))
DEV = os.environ.get("EVAL_DEVICE", "cpu")
BATCH = int(os.environ.get("EVAL_BATCH", "512"))
DRIFT_FLOOR = 0.0739     # points by position; lens/runs/DRIFT-NUMBERS.txt (measured 2026-10-25 17:2x)

_DEFAULT = [
    ("twin    ct_d256_full169_fix3 (3 ep, stored 35.2632%)", "lens/runs/chat/ct_d256_full169_fix3_best.npz"),
    ("H2      h2_noconcept_169_3   (3 ep, stored 35.7220%)", "lens/runs/chat/h2_noconcept_169_3_best.npz"),
    ("H5L     ct_d256_full169_6ep  (6 ep, stored 36.6878%)", "lens/runs/chat/ct_d256_full169_6ep_best.npz"),
    ("H2L     h2l_noconcept_169_6ep(6 ep, see its own row)", "lens/runs/chat/h2l_noconcept_169_6ep_best.npz"),
]


def log(s=""):
    print(s, flush=True)


def ckpt_list():
    spec = os.environ.get("CKPT_LIST", "").strip()
    if not spec:
        return _DEFAULT
    if spec.lower() == "none":
        # population-only pass: measures the held-out column's population (P108a(i), P108b) and loads no model
        return []
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.basename(label), label
        if not os.path.isabs(path):
            path = os.path.join(P, path)
        out.append((label.strip(), path))
    return out


def load(ckpt):
    z = np.load(ckpt, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
    T.set_device(DEV)
    net = LensNet(meta["V"], concepts, d=meta["D"], window=meta["W"], seed=meta["seed"],
                  lam_concept=meta["lam_concept"], lam_act=meta["lam_act"],
                  attention=bool(meta.get("attention", False)), numeric=bool(meta.get("numeric", False)),
                  direct=bool(meta.get("direct", False)), direct_rank=int(meta.get("direct_rank", 16)),
                  direct_mode=str(meta.get("direct_mode", "lowrank")))
    net.device_name = DEV
    for p in net.parameters():
        p.data = B.to_dev(z[p.name].astype(np.float32))
    return net, meta, z


def asst_flags(toks):
    """Mirror of the trainer's is_asst walk (train_chat.py ~lines 553-588), is_asst only.

    The trainer compares `toks[k].lower()` against the markers and `toks[k+1].lower()` against the role, so the
    same .lower() is applied here; the corpus vocab is already lowercased except the two control tokens.
    """
    fa = np.zeros(len(toks), dtype=bool)
    in_asst = False
    k = 0
    n = len(toks)
    while k < n:
        t = toks[k].lower()
        if t == "<|im_start|>" and k + 1 < n:
            in_asst = toks[k + 1].lower() == "assistant"
            k += 2
            continue
        if t == "<|im_end|>":
            k += 1
            continue
        if t == "assistant" and k + 1 < n and toks[k + 1] == ":":
            in_asst = True
            k += 2
            continue
        if t == "human" and k + 1 < n and toks[k + 1] == ":":
            in_asst = False
            k += 2
            continue
        if in_asst:
            fa[k] = True
        k += 1
    return fa


def main():
    t_start = time.time()
    dd = os.environ["CHAT_DATA_DIR"]
    meta_f = os.path.join(dd, "meta_v12000.json")
    log("=" * 110)
    log(f"DIALOGUE vs MIXED HELD-OUT EVAL  {time.strftime('%F %H:%M')}"
        f"   scope={os.environ.get('DM_SCOPE', 'self-wake')}")
    log("predictions P108a-h: lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt (written before this file existed)")
    log(f"corpus (read-only): {dd}")
    log(f"device={DEV}  window={W}  batch={BATCH}  drift floor (by position)={DRIFT_FLOOR} pts "
        f"[lens/runs/DRIFT-NUMBERS.txt]")
    log("=" * 110)

    if not os.path.exists(meta_f):
        log(f"REFUSING: no meta file at {meta_f}")
        return 2
    mmeta = json.load(open(meta_f))
    per_src = mmeta["per_source_corpus"]
    ext_key = [k for k in per_src if "ext_honest" in k]
    if not ext_key:
        log(f"REFUSING: cannot find the ext_honest group in {meta_f}: {list(per_src)}")
        return 2
    n_ext = int(per_src[ext_key[0]]["conversations"])
    log(f"group boundary from {os.path.basename(meta_f)}: ext_honest = conversations 0..{n_ext - 1} "
        f"({n_ext:,} convs, HUMAN DIALOGUE)   kn2 = {n_ext}..{int(mmeta['conversations']) - 1} "
        f"({int(mmeta['conversations']) - n_ext:,} convs, maths/textbook/code)")
    log(f"P105a (contiguity of the two groups) was checked in lens/runs/CORPUS-MIX-AUDIT.txt; this script reads "
        f"the boundary from the same meta file that audit used.")

    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    bnd = np.asarray(bounds, dtype=np.int64)
    Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)
    log(f"  val windows (W={W}, stride {W // 2}, no crossing): {len(Xva):,}   "
        f"[the trainer's own call: chat_data.make_windows(..., which=1)]")

    conv_of = np.searchsorted(bnd, Sva, side="right") - 1
    if not (np.all(bnd[conv_of] <= Sva) and np.all(Sva < bnd[conv_of + 1])):
        log("REFUSING: window->conversation mapping failed a boundary check")
        return 2
    is_ext = conv_of < n_ext

    # ---- assistant mask (by position) and the dialogue/kn2 split, on the trainer's own target positions ----
    mask = np.zeros(len(Sva), dtype=bool)
    tok_ext = tok_kn2 = 0
    convs = np.unique(conv_of)
    start = 0
    for j in convs:
        end = int(np.searchsorted(conv_of, j, side="right"))
        if end <= start:
            start = end
            continue
        a0, b0 = int(bnd[j]), int(bnd[j + 1])
        toks = [vocab[int(t)] for t in np.asarray(ids[a0:b0])]
        if len(toks) != b0 - a0:
            log(f"REFUSING: conversation {j} read {len(toks)} tokens, expected {b0 - a0}")
            return 2
        fa = asst_flags(toks)
        rel = Sva[start:end] - a0
        mask[start:end] = fa[rel + W]
        msk = mask[start:end]
        if is_ext[start]:
            tok_ext += int(msk.sum())
        else:
            tok_kn2 += int(msk.sum())
        start = end
    n_asst = int(mask.sum())
    n_all = int(len(Sva))
    log("")
    log("POPULATION OF THE HELD-OUT COLUMN (the trainer's own assistant-by-position positions, n = val_asst_n):")
    log(f"  val windows (all positions, the 'VAL acc' column):        {n_all:,}")
    log(f"  ASSISTANT-by-position targets (the 'ASST' column):        {n_asst:,}")
    log(f"    of which DIALOGUE  (ext_honest):                        {tok_ext:,}  "
        f"({tok_ext / max(n_asst, 1) * 100:.2f}%)")
    log(f"    of which KN2 (maths/textbook/code):                     {tok_kn2:,}  "
        f"({tok_kn2 / max(n_asst, 1) * 100:.2f}%)")
    p108a_n = (n_asst == 58191)
    log(f"  P108a(i) CONTROL: n must read exactly 58,191 (every stored epoch row in "
        f"lens/runs/chat/*_history.jsonl) -> measured {n_asst:,} -> "
        f"{'PASS' if p108a_n else 'FAIL'}")
    share = tok_ext / max(n_asst, 1) * 100.0
    log(f"  P108b (dialogue share 15-30%): measured {share:.2f}% -> "
        f"{'PASS' if 15.0 <= share <= 30.0 else ('FAIL (outside 15-30%)' if 10.0 <= share <= 40.0 else 'FAIL (outside even the 10-40% outer band)')}")

    log("")
    log("=" * 110)
    log("PER-CHECKPOINT COLUMNS  (accuracy / perplexity / n)")
    log("=" * 110)

    results = []
    for label, ckpt in ckpt_list():
        if not os.path.exists(ckpt):
            log("")
            log(f"SKIPPED (no such file): {ckpt}")
            continue
        net, meta, z = load(ckpt)
        log("")
        log(f"CHECKPOINT {label}")
        log(f"  file: {os.path.relpath(ckpt, P)}  ({os.path.getsize(ckpt):,} B, "
            f"{time.strftime('%F %H:%M', time.localtime(os.path.getmtime(ckpt)))})")
        log(f"  meta: V={meta['V']} D={meta['D']} W={meta['W']} seed={meta['seed']} "
            f"direct={meta.get('direct')} direct_mode={meta.get('direct_mode')} "
            f"attention={meta.get('attention')}")
        if int(meta["W"]) != W:
            log(f"  REFUSED: this checkpoint's window is {int(meta['W'])} and the val window set here is "
                f"W={W}; a window-{int(meta['W'])} ship cannot be measured with W={W} input "
                f"(the pad-free instrument makes the same refusal). NOT measured.")
            del net
            continue

        def blank():
            return dict(ce=0.0, ce_ppl=0.0, corr=0, n=0)

        st = {"all": blank(), "asst": blank(), "ext": blank(), "kn2": blank()}
        t0 = time.time()
        with np.errstate(over="ignore"):
            for s0 in range(0, len(Xva), BATCH):
                s1 = min(s0 + BATCH, len(Xva))
                idx = np.arange(s0, s1)
                lg = np.asarray(net.forward_arrays(Xva[idx])["logits"], dtype=np.float64)
                y = Yva[idx]
                mx = lg.max(axis=1, keepdims=True)
                logp = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
                ce = -logp[np.arange(len(idx)), y]
                pred = lg.argmax(axis=1)

                def add(k, sel):
                    if sel is None:
                        sel = np.ones(len(idx), dtype=bool)
                    nn = int(sel.sum())
                    if nn == 0:
                        return
                    st[k]["ce"] += float(ce[sel].sum())
                    st[k]["corr"] += int((pred[sel] == y[sel]).sum())
                    st[k]["n"] += nn

                add("all", None)
                add("asst", mask[idx])
                add("ext", mask[idx] & is_ext[idx])
                add("kn2", mask[idx] & (~is_ext[idx]))
        secs = time.time() - t0

        def line(name, k):
            r = st[k]
            if r["n"] == 0:
                return f"    {name:<34} n=0"
            acc = r["corr"] / r["n"] * 100.0
            ppl = float(np.exp(r["ce"] / r["n"]))
            return f"    {name:<34} acc {acc:8.4f}%   ppl {ppl:9.3f}   n {r['n']:,}"

        log(line("ALL val windows (no mask)", "all"))
        log(line("ASST by position, ALL groups", "asst"))
        log(line("ASST by position, DIALOGUE only", "ext"))
        log(line("ASST by position, KN2 only", "kn2"))
        log(f"    ({secs:.0f}s on {DEV})")

        res = {k: (st[k]["corr"] / st[k]["n"] * 100.0 if st[k]["n"] else float("nan")) for k in st}
        res["n"] = {k: st[k]["n"] for k in st}
        res["label"] = label
        res["file"] = os.path.relpath(ckpt, P)
        results.append(res)

        # --- the stored-number control (P108a ii): the trainer's own epoch rows are the reference -------
        hist = os.path.join(P, "lens/runs/chat", os.path.basename(ckpt).replace("_best.npz", "_history.jsonl"))
        if os.path.exists(hist):
            rows = [json.loads(x) for x in open(hist) if x.strip()]
            best = max((r for r in rows if r.get("val_asst_acc") is not None),
                       key=lambda r: r["val_asst_acc"], default=None)
            if best is not None:
                d_pts = res["asst"] - best["val_asst_acc"] * 100.0
                log(f"  P108a(ii) CONTROL vs the trainer's own best epoch row ({os.path.basename(hist)}, "
                    f"epoch {best['epoch']}):")
                log(f"      stored val_asst_acc {best['val_asst_acc'] * 100:.4f}%   this instrument "
                    f"{res['asst']:.4f}%   |diff| {abs(d_pts):.4f} pts  (floor {DRIFT_FLOOR}) -> "
                    f"{'PASS' if abs(d_pts) <= DRIFT_FLOOR else 'FAIL'}")
            else:
                log(f"  P108a(ii) CONTROL: {os.path.basename(hist)} has no val_asst_acc row -> NOT MEASURABLE")
        else:
            log(f"  P108a(ii) CONTROL: no history file at {os.path.relpath(hist, P)} -> NOT MEASURABLE "
                f"(the arm may still be training; run this again when it finishes)")
        del net

    # ---- P108c-f: the comparisons, with both values printed beside every verdict ----
    log("")
    log("=" * 110)
    log("VERDICTS, WITH THE COMPARED VALUES PRINTED BESIDE EACH ONE")
    log("=" * 110)
    if len(results) < 1:
        log("  no checkpoint was measured: P108c-f are NOT DECIDED this pass.")
    else:
        # GRADE AGAINST THE DECLARED POINT BAND, and print the falsifier band beside it.
        # Defect fixed 2026-10-25 19:3x (BUGS.md [BUG-P108C-LEDGER-LINE-GRADES-AGAINST-THE-FALSIFIER-BAND]): the
        # first run of this file labelled P108c "PASS" using the FALSIFIER band (0.5-20 points) while the
        # prediction declared a 3-10 POINT band, and the measured gaps (11.58-11.76) missed it. The numbers were
        # right; the label was computed against the wrong band. This line now names both.
        ok = True
        for r in results:
            gap = r["asst"] - r["ext"]
            declared = (3.0 <= gap <= 10.0)
            falsified = not (r["ext"] < r["asst"] and 0.5 <= gap <= 20.0)
            v = ("PASS (inside the declared 3-10 pt band)" if declared else
                 ("BAND MISS (direction held; falsifier band 0.5-20 not fired)" if not falsified else
                  "FAIL (falsifier fired)"))
            ok &= declared
            log(f"  P108c {r['label'].split()[0]:<8} dialogue {r['ext']:.4f}%   mixed {r['asst']:.4f}%   "
                f"gap {gap:+.4f} pts -> {v}")
        log(f"  P108c overall: {'PASS' if ok else 'NOT PASSED (see the band named on each line)'} "
            f"(declared band 3-10 points; falsifier band 0.5-20 AND dialogue below mixed)")
        ok = True
        for r in results:
            gap = r["kn2"] - r["asst"]
            declared = (1.0 <= gap <= 8.0)
            falsified = not (r["kn2"] > r["asst"] and 0.5 <= gap <= 15.0)
            v = ("PASS (inside the declared 1-8 pt band)" if declared else
                 ("BAND MISS (direction held; falsifier band 0.5-15 not fired)" if not falsified else
                  "FAIL (falsifier fired)"))
            ok &= declared
            log(f"  P108d {r['label'].split()[0]:<8} kn2 {r['kn2']:.4f}%   mixed {r['asst']:.4f}%   "
                f"gap {gap:+.4f} pts -> {v}")
        log(f"  P108d overall: {'PASS' if ok else 'NOT PASSED (see the band named on each line)'} "
            f"(declared band 1-8 points; falsifier band 0.5-15 AND kn2 above mixed)")

        by_mixed = sorted(results, key=lambda r: -r["asst"])
        by_ext = sorted(results, key=lambda r: -r["ext"])
        log("  ranking on the mixed column : " + " > ".join(
            f"{r['label'].split()[0]} {r['asst']:.2f}" for r in by_mixed))
        log("  ranking on dialogue-only    : " + " > ".join(
            f"{r['label'].split()[0]} {r['ext']:.2f}" for r in by_ext))
        lead = by_mixed[0]
        worst_diff = max(r["ext"] - lead["ext"] for r in results)
        log(f"  P108e: mixed-column leader {lead['label'].split()[0]} dialogue {lead['ext']:.4f}%  vs best "
            f"dialogue {by_ext[0]['ext']:.4f}% -> deficit {worst_diff:+.4f} pts -> "
            f"{'PASS' if worst_diff <= 1.0 else 'FAIL'} (predicted: no checkpoint beats the mixed leader on "
            f"dialogue by more than 1.0 pt)")
        spread = max(r["ext"] for r in results) - min(r["ext"] for r in results)
        spread_mixed = max(r["asst"] for r in results) - min(r["asst"] for r in results)
        log(f"  P108f: dialogue-only spread {spread:.4f} pts across {len(results)} checkpoints "
            f"(mixed-column spread {spread_mixed:.4f} pts) -> "
            f"{'PASS' if spread <= 2.0 else ('MISS' if spread <= 3.0 else 'FAIL')} "
            f"(predicted <= 2.0)")
        log(f"  P108g: no generation was run by this instrument; the ship decider (0/6, "
            f"lens/runs/PADFREE-TALK-*.txt) is untouched.")

    log("")
    log(f"TOTAL {time.time() - t_start:.0f}s   every input was read-only; this file wrote nothing but stdout.")
    log("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
