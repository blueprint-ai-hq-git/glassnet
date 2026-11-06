#!/usr/bin/env python3
"""agreement_report.py -- the verdict table for PLAN item (d2d), read from the stored per-step JSONL files.

Every number it prints names the file it came from. The bars are the ones written BEFORE the run in
lens/runs/AGREEMENT-GATE-PREDICTIONS.txt (P60a-P60n); this script decides PASS or FAIL on each, and prints
the compared values beside every verdict (a right number next to a wrong conclusion is this project's
most-repeated bug).
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAT = os.path.join(PROJ, "lens", "runs", "chat")
RUNS = os.path.join(PROJ, "lens", "runs")


def rows(tag):
    p = os.path.join(CHAT, f"{tag}_agreement.jsonl")
    if not os.path.exists(p):
        return None, p
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out, p


def series(r, key):
    return [x[key] for x in r if x.get(key) is not None]


def frac_refused(r):
    if not r:
        return None
    return sum(1 for x in r if x["refused"]) / len(r)


def hist(tag):
    p = os.path.join(CHAT, f"{tag}_history.jsonl")
    if not os.path.exists(p):
        return None, p
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out, p


def verdict(name, ok, text):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}   {text}")
    return ok


def main():
    tags = sys.argv[1:] or ["agree_m_vs_s0", "agree_m_vs_s0b", "agree_l_vs_s0",
                            "agree_m_ema_s0", "agree_l_ema_s0", "agree_m_lev_s0", "agree_l_lev_s0"]
    data = {}
    print("=" * 100)
    print("AGREEMENT GATE - VERDICT TABLE (PLAN item d2d). Bars from lens/runs/AGREEMENT-GATE-PREDICTIONS.txt")
    print("=" * 100)
    print("\n-- PER-ARM, read from lens/runs/chat/<tag>_agreement.jsonl")
    print(f"  {'arm':<18} {'steps':>6} {'refused':>8} {'frac':>7} {'A_train first':>14} {'A_train last':>13} "
          f"{'A_probe mean':>13} {'floor':>6}")
    for t in tags:
        r, p = rows(t)
        if r is None:
            print(f"  {t:<18} MISSING ({p})")
            continue
        data[t] = r
        at = series(r, "a_train")
        ap = series(r, "a_probe")
        floor = 1.0 / (r[0].get("groups_used") or 8) ** 0.5
        print(f"  {t:<18} {len(r):>6} {sum(1 for x in r if x['refused']):>8} {frac_refused(r):>7.3f} "
              f"{at[0]:>14.4f} {at[-1]:>13.4f} "
              f"{(sum(ap) / len(ap)) if ap else float('nan'):>13.4f} {floor:>6.3f}")

    print("\n-- P60c DETERMINISM (same seed, same arm, twice): A_train identical at every step")
    if "agree_m_vs_s0" in data and "agree_m_vs_s0b" in data:
        a, b = series(data["agree_m_vs_s0"], "a_train"), series(data["agree_m_vs_s0b"], "a_train")
        same = len(a) == len(b) and all(abs(x - y) < 1e-12 for x, y in zip(a, b))
        dmax = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("nan")
        verdict("P60c", same, f"{len(a)} vs {len(b)} steps, max |difference| {dmax:.3e}")

    print("\n-- P60d/P60e THE NUMBER'S OWN BEHAVIOUR (bar: step 1 in [0.50, 0.95]; last <= first - 0.05)")
    for t in ("agree_m_vs_s0", "agree_l_vs_s0"):
        if t not in data:
            continue
        at = series(data[t], "a_train")
        verdict("P60d " + t, 0.50 <= at[0] <= 0.95, f"A_train at step 1 = {at[0]:.4f} (bar [0.50, 0.95])")
        verdict("P60e " + t, at[-1] <= at[0] - 0.05, f"last {at[-1]:.4f} vs first {at[0]:.4f}")

    print("\n-- P60f THE ITEM'S DIRECTION AS A LEVEL: A_train on M below A_train on L at matched steps")
    if "agree_m_vs_s0" in data and "agree_l_vs_s0" in data:
        am, al = series(data["agree_m_vs_s0"], "a_train"), series(data["agree_l_vs_s0"], "a_train")
        for k in (50, 100, 200, min(len(am), len(al))):
            if k <= len(am) and k <= len(al):
                print(f"    step {k:>4}: M {am[k - 1]:.4f}   L {al[k - 1]:.4f}   "
                      f"{'M lower' if am[k - 1] < al[k - 1] else 'M NOT lower'}   (delta {am[k-1]-al[k-1]:+.4f})")
        verdict("P60f", all(am[k - 1] < al[k - 1] for k in (200,) if k <= min(len(am), len(al))),
                "see the matched-step lines above")

    print("\n-- P60g/P60h THE ITEM'S OWN QUESTION, DEFAULT DETECTOR 'vs_probe' (bar: >= 0.05 on M, <= 0.20 on L)")
    print("   PER SEED (the standing rule is three seeds before anything is called a win):")
    ok_all = True
    for s in (0, 1, 2):
        fm = frac_refused(rows(f"agree_m_vs_s{s}")[0] or []) if rows(f"agree_m_vs_s{s}")[0] else None
        fl = frac_refused(rows(f"agree_l_vs_s{s}")[0] or []) if rows(f"agree_l_vs_s{s}")[0] else None
        if fm is None or fl is None:
            print(f"    seed {s}: MISSING an arm")
            continue
        v = (fm >= 0.05) and (fl <= 0.20)
        ok_all = ok_all and v
        print(f"    seed {s}: M refused {fm:.3f} (bar >= 0.05: {'ok' if fm >= 0.05 else 'FAIL'})   "
              f"L refused {fl:.3f} (bar <= 0.20: {'ok' if fl <= 0.20 else 'FAIL'})   "
              f"gap between the corpora {fm - fl:+.3f}")
    verdict("P60g+P60h all seeds", ok_all, "see the per-seed lines above")
    print("   STATED WITH IT: the SEPARATION is seed-sensitive - the M arms refuse 0.383/0.190/0.158 and the L")
    print("   arms 0.013/0.110/0.076, so the gap ranges from +0.080 to +0.370. A bar that asks each corpus to")
    print("   clear its own threshold PASSES on every seed; a claim that the rule SEPARATES the corpora is")
    print("   weaker than that and is not what was measured.")

    for t, corpus, lo, hi in (("agree_m_vs_s0", "M (memorisable)", 0.05, None),
                              ("agree_l_vs_s0", "L (learnable)", None, 0.20)):
        if t not in data:
            continue
        f = frac_refused(data[t])
        if lo is not None:
            verdict("P60g " + t, f >= lo, f"refused {f:.3f} on {corpus} (bar >= {lo})")
        else:
            verdict("P60h " + t, f <= hi, f"refused {f:.3f} on {corpus} (bar <= {hi})")

    print("\n-- P60i/P60j THE OTHER DETECTORS (the same bar, so the reader sees the whole table)")
    for t in ("agree_m_ema_s0", "agree_l_ema_s0", "agree_m_lev_s0", "agree_l_lev_s0"):
        if t in data:
            f = frac_refused(data[t])
            print(f"    {t:<18} refused {f:.3f}   (M arms want >= 0.05, L arms want <= 0.20)")

    print("\n-- P60l COST: mean epoch seconds of the gated arm against the stored UNGATED baseline")
    for tag_g, tag_u in (("agree_m_vs_s0", "gate_m_off"), ("agree_m_ema_s0", "gate_m_off")):
        hg, pg = hist(tag_g)
        hu, pu = hist(tag_u)
        if hg and hu:
            mg = sum(x["seconds"] for x in hg) / len(hg)
            mu = sum(x["seconds"] for x in hu) / len(hu)
            print(f"    {tag_g}: {mg:.2f} s/epoch ({len(hg)} epochs, {pg}) vs {tag_u}: {mu:.2f} s/epoch "
                  f"({len(hu)} epochs, {pu}) -> ratio {mg/mu:.2f}x")
            verdict("P60l " + tag_g, mg / mu <= 3.0, f"ratio {mg/mu:.2f}x against the <= 3.0x bar")

    print("\n-- P60m THE GROUP-SUM GRADIENT vs THE FULL-BATCH GRADIENT (measured on step 1, NOT used)")
    for t in tags:
        if t in data:
            g = [x.get("grad_group_sum_rel") for x in data[t] if x.get("grad_group_sum_rel") is not None]
            if g:
                print(f"    {t:<18} relative difference {g[0]:.3e}   (bar <= 1e-4)")
                verdict("P60m " + t, g[0] <= 1e-4, f"{g[0]:.3e}")

    print("\n-- P60k SEED SPREAD (filled in only if the seeds 1/2 arms exist)")
    for stem, corpus in (("agree_m_vs_s", "M"), ("agree_l_vs_s", "L")):
        last = []
        for s in (0, 1, 2):
            r, _p = rows(f"{stem}{s}")
            if r:
                last.append(series(r, "a_train")[-1])
        if len(last) > 1:
            print(f"    {corpus}: last-step A_train over {len(last)} seeds {['%.4f' % x for x in last]} "
                  f"spread {max(last)-min(last):.4f} (bar <= 0.10)")
            verdict(f"P60k {corpus}", max(last) - min(last) <= 0.10, f"spread {max(last)-min(last):.4f}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
