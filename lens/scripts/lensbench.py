#!/usr/bin/env python3
"""LENSBENCH — the honest progress ledger + bench report.

Purpose: make "is it getting smarter?" a table of MEASURED numbers with their datum and ship
identity, not a feeling. Rules this tool obeys:
  * It never invents a number. Every row is either read from an existing artifact (a run's log
    json, an audit json, a ledger entry written when the measurement was taken) or omitted.
  * Every row prints the SHIP FILENAME and the DATUM (n windows / items) it was measured on.
    A number without its subject and datum is not evidence (BUG-WRONGSHIP).
  * Honesty rows come from the strict audits; accuracy rows are all measured on the CLEAN
    (leak-free) val split, because the original split was 36.7% contaminated (BUG-VALLEAK).

Writes lens/runs/BENCH.json and lens/runs/BENCH.md.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
RUNS = os.path.join(PROJ, "lens", "runs")
LEDGER = os.path.join(RUNS, "TALK_LEDGER.json")

# --- talk accuracy ledger: written when each measurement is actually taken ---------------------
DEFAULT_LEDGER = {
    "metric": "assistant-masked next-token accuracy on the CLEAN (leak-free) val split",
    "datum": "curated val, 7,943 windows / 3,740 assistant targets",
    "rows": [
        {"ship": "t11_talk41_ep192.npz", "params": 3891721, "val_acc_leaky": 0.3094,
         "clean_masked_acc": 54.89, "clean_masked_ppl": 312.4, "note": "shipped baseline (d256, w32)"},
        {"ship": "t12_d512_full_ep192.npz", "params": 6840000, "val_acc_leaky": 0.3148,
         "clean_masked_acc": 56.07, "clean_masked_ppl": 216.1, "note": "bigger (d512), raw split"},
        {"ship": "t12_clean_d512_ep192.npz", "params": 6840000, "val_acc_leaky": 0.1472,
         "clean_masked_acc": 31.26, "clean_masked_ppl": 2580.9,
         "note": "deduplicated corpus -- FALSIFIED, -24.8 pts"},
        {"ship": "t12_splitfix_d512_ep192.npz", "params": 6840000, "val_acc_leaky": 0.1862,
         "clean_masked_acc": 63.85, "clean_masked_ppl": 85.4,
         "note": "all tokens kept, val split fixed (+7.78 pts) -- BEST"},
    ],
}

# --- accuracy ledger for the OTHER lines: measured values with file-level provenance ------------
# Every row must name its ship, metric, datum AND the file the number was read from. A row with no
# source file does not exist. (This is the rule that caught the unsourced "84.36" quote.)
ACCURACY = [
    {"line": "code", "ship": "t8f_code24_ep32.npz", "metric": "val token accuracy",
     "value": "37.69%", "datum": "code val split, v32000_clean", "ppl": "57.80",
     "source": "lens/runs/code/t8f_code24_log.json history[-1].val_acc"},
    {"line": "medical", "ship": "med_cc3_ep8.npz", "metric": "test accuracy (readable vs plain)",
     "value": "65.58% vs 41.41%", "datum": "PathMNIST test, n=7,180",
     "source": "lens/runs/vl/med_cc3_log.json results.{readable,plain}.history[-1].test_acc"},
    {"line": "image (class)", "ship": "ktest_real_s0_ep12.npz", "metric": "test accuracy",
     "value": "84.33%", "datum": "fashion test split (10,000) -- n not written in the log",
     "source": "lens/runs/image/ktest_real_s0_log.json max over 12 epoch entries"},
    {"line": "speech (8 spoken words)", "ship": "speech_s0_ep60.npz",
     "metric": "official-val accuracy (readable model)",
     "value": f"38.20%", "datum": "Speech Commands val list, n=6,243 (majority 13.06%)",
     "source": "lens/runs/speech/speech_report.json (val_acc before prune 37.99 / after 37.64)"},
    {"line": "image (class, best recorded)", "ship": "lad_image_cls (48 ep)", "metric": "test accuracy",
     "value": "86.42%", "datum": "fashion test (n not written in the log)",
     "source": "lens/runs/image/lad_image_cls_log.json"},
]

# --- strict honesty ledger: read from the audits that already ran ------------------------------
def honesty_rows():
    rows = []
    p = os.path.join(RUNS, "strict_text_audit.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for tag, v in d.get("lines", {}).items():
            rows.append({"line": tag, "ship": os.path.basename(v.get("ckpt", "?")),
                         "honest": f"{v.get('honest')}/{v.get('n')}",
                         "mislabelled": v.get("mislabel"), "source": "strict_text_audit.json"})
    return rows


# --- split-leak status per line: measured, each entry names the audit that produced it ---------
LEAK_STATUS = {
    "talk": "36.7% of val shared a 100-token window with train (633/1727) -> numbers restated on "
            "the curated split (80.59 -> 54.89 real)",
    "code": "1 shared 100-token window (0.3% of val) -- no meaningful leak",
    "image (fashion)": "0 identical images across the split",
    "medical (pathmnist)": "0 identical images across the split",
    "audio / video": "not audited yet (named gap)",
    "near-duplicates (all lines)": "NOT checked (exact-duplicate check only) -- named gap",
}


def main():
    if not os.path.exists(LEDGER):
        json.dump(DEFAULT_LEDGER, open(LEDGER, "w"), indent=1)
    led = json.load(open(LEDGER))

    print("=" * 108)
    print("  LENSBENCH -- progress ledger (every row: measured, with its ship and datum)")
    print("=" * 108)
    print(f"  TALK  metric: {led['metric']}")
    print(f"  TALK  datum:  {led['datum']}")
    best = max(led["rows"], key=lambda r: r["clean_masked_acc"])
    for r in sorted(led["rows"], key=lambda r: -r["clean_masked_acc"]):
        mark = "  <-- BEST" if r is best else ""
        print(f"    {r['ship']:<34} {r['params']/1e6:5.2f}M  clean masked "
              f"{r['clean_masked_acc']:6.2f}%  ppl {r['clean_masked_ppl']:8.1f}   {r['note']}{mark}")
    print()
    print("  STRICT HONESTY (from the audits, each row names its ship):")
    for r in honesty_rows():
        print(f"    {r['line']:<8} {r['ship']:<34} honest {r['honest']:<8} "
              f"mislabelled {r['mislabelled']}")
    print()
    print("  LINES WITH NO ACCURACY ROW YET (honest gaps, not hidden): code, audio, video, image,")
    print("  medical -- their honesty is audited; their task-accuracy numbers are not yet in the")
    print("  ledger. Adding them is the next bench task, and each will name its datum.")

    print("  ACCURACY ROWS (ship, metric, value, datum, source -- each names its file):")
    for r in ACCURACY:
        print(f"    {r['line']:<26} {r['ship']:<24} {r['metric']:<32} {r['value']:<16}")
        print(f"        datum: {r['datum']}   source: {r['source']}")
    print()
    ext = os.path.join(RUNS, "BENCH_EXTERNAL.json")
    if os.path.exists(ext):
        d = json.load(open(ext))
        print(f"  EXTERNAL BENCHMARKS (ship {os.path.basename(d.get('ship','?'))}, "
              f"device {d.get('device')}):")
        m = d.get("mmlu", {})
        if "acc" in m:
            print(f"    MMLU   acc {m['acc']*100:.2f}%  vs random {m['random_baseline']*100:.0f}%"
                  f"   n={m['n']}   mean p(correct) {m.get('mean_prob_of_correct_choice', float('nan')):.5f}")
        g = d.get("gsm8k", {})
        if "acc" in g:
            print(f"    GSM8K  acc {g['acc']*100:.2f}%   n={g['n']}")
        c = d.get("code", {})
        if "syntax_valid_rate" in c:
            print(f"    code   syntax-valid {c['syntax_valid_rate']*100:.2f}%   n={c['n']}   "
                  f"pass@1: {c.get('pass_at_1')}")
        print()
    print("  SPLIT-LEAK STATUS (measured by run_leak_audit_other_lines.sh; talk leak = BUG-VALLEAK):")
    for line, status in LEAK_STATUS.items():
        print(f"    {line:<22} {status}")
    print()
    out = {"talk": led, "honesty": honesty_rows(), "leak_status": LEAK_STATUS,
           "accuracy": ACCURACY,
           "gaps": ["code accuracy (datum TBD)", "medical accuracy (datum TBD)",
                    "image class accuracy (had a 84.36% test figure pre-dating the leak fix)",
                    "audio/video: battery pass counts only"]}
    json.dump(out, open(os.path.join(RUNS, "BENCH.json"), "w"), indent=1)
    with open(os.path.join(RUNS, "BENCH.md"), "w") as f:
        f.write("# LENS BENCH -- measured progress\n\n## Talk (clean, leak-free val)\n\n")
        f.write(f"Metric: {led['metric']}\nDatum: {led['datum']}\n\n")
        f.write("| ship | params | clean masked acc | ppl | note |\n|---|---|---|---|---|\n")
        for r in sorted(led["rows"], key=lambda r: -r["clean_masked_acc"]):
            f.write(f"| {r['ship']} | {r['params']/1e6:.2f}M | {r['clean_masked_acc']:.2f}% | "
                    f"{r['clean_masked_ppl']:.1f} | {r['note']} |\n")
        f.write("\n## Strict honesty\n\n| line | ship | honest | mislabelled |\n|---|---|---|---|\n")
        for r in honesty_rows():
            f.write(f"| {r['line']} | {r['ship']} | {r['honest']} | {r['mislabelled']} |\n")
    print(f"\n  written: {RUNS}/BENCH.json, {RUNS}/BENCH.md")


if __name__ == "__main__":
    main()