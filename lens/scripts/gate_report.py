"""gate_report.py - the verdict table for PLAN item (d2b): the memorisation rule wired into the SHIPPED
trainer, and the question that decides whether the wiring is real - does the gate FIRE on a corpus that can
be memorised, in the trainer that ships, on real text?

Predictions were written BEFORE anything ran: lens/runs/TRAIN-GATES-PREDICTIONS.txt (P58a-P58j). This script
only reads the files the runs wrote; it computes nothing that a run did not measure, and it prints the
compared values NEXT TO every verdict, because a right number given a wrong verdict is the failure mode this
project has hit before (BUGS.md).
"""
import glob
import hashlib
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "lens", "runs", "chat")
os.chdir(PROJ)

ARMS = ["gate_m_off", "gate_m_on", "gate_m_on2", "gate_l_on", "gate_m_relgrad"]


def last_row(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[-1] if rows else None


def summary(arm):
    p = os.path.join(CH, f"{arm}_gates_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def jsonl_rows(arm):
    p = os.path.join(CH, f"{arm}_gates.jsonl")
    if not os.path.exists(p):
        return 0
    return sum(1 for l in open(p) if l.strip())


def exits():
    out = {}
    p = "lens/runs/GATE-TEST.txt"
    if os.path.exists(p):
        for line in open(p):
            if " EXIT=" in line:
                out[line.split(" EXIT=")[0].strip()] = int(line.split(" EXIT=")[1].strip())
    return out


print("=" * 100)
print("PLAN item (d2b): THE MEMORISATION RULE (lens/unified.py GapGate) INSIDE THE SHIPPED TRAINER")
print("predictions: lens/runs/TRAIN-GATES-PREDICTIONS.txt   raw output: lens/runs/GATE-TEST.txt")
print("=" * 100)

print("\nTHE ARMS (last epoch of each; every number read from lens/runs/chat/<arm>_history.jsonl)")
print(f"  {'arm':<16} {'epochs':>6} {'train_loss':>10} {'val_ppl':>10} {'val_acc':>8} "
      f"{'refused':>9} {'refused_frac':>12} {'probe_first':>11} {'probe_last':>10}")
rows = {}
for arm in ARMS:
    r = last_row(os.path.join(CH, f"{arm}_history.jsonl"))
    s = summary(arm)
    n = sum(1 for _ in open(os.path.join(CH, f"{arm}_history.jsonl"))) if r else 0
    rows[arm] = {"last": r, "summ": s, "epochs": n}
    if r is None:
        print(f"  {arm:<16} {'-':>6}   (no history file)")
        continue
    rf = f"{s['refused_fraction']:.3f}" if s else "-"
    pf = f"{s['probe_loss_first']:.4f}" if s else "-"
    pl = f"{s['probe_loss_last']:.4f}" if s else "-"
    print(f"  {arm:<16} {n:>6} {r['train_loss']:>10.4f} {r['val_ppl']:>10.2f} {r['val_acc']*100:>7.2f}% "
          f"{(s['steps_refused'] if s else 0):>9} {rf:>12} {pf:>11} {pl:>10}")

ex = exits()
print(f"\n  exit codes (lens/runs/GATE-TEST.txt): {ex}")

print("\nVERDICTS - binary, with the compared values beside each one")
smoke = open("lens/runs/GATE-SMOKE.txt").read()

m_off, m_on, m_on2, l_on, m_rg = (rows[a] for a in ARMS)
s_on, s_on2, s_l, s_rg = m_on["summ"], m_on2["summ"], l_on["summ"], m_rg["summ"]


def verdict(tag, ok, text):
    print(f"  {tag} {'PASS' if ok else 'FAIL'}: {text}")


# P58a - the default path is unchanged
ab = "A vs B: IDENTICAL" in smoke
verdict("P58a", ab, "the edited trainer with no new flags vs the pre-change copy, whole epoch lines: "
                    f"{'0 differing' if ab else 'DIFFERING - see lens/runs/GATE-SMOKE.txt'}")

# P58c - the control
c_nums = "A vs C (numbers only): IDENTICAL" in smoke
c_zero = "GATE RECORD: 0 of" in smoke
verdict("P58c", c_nums and c_zero,
        f"control (--gate-variant off): refusals {'0' if c_zero else 'NON-ZERO'} and epoch lines with the "
        f"wall-clock column stripped {'identical' if c_nums else 'DIFFERING'}; the WHOLE-line comparison is "
        f"{'identical' if 'A vs C: IDENTICAL' in smoke else 'DIFFERENT on the timing column alone'}")

# P58b - THE ITEM: the gate fires on the memorisable corpus
if s_on:
    frac = s_on["refused_fraction"]
    verdict("P58b", frac >= 0.05,
            f"refused fraction on the corpus that CAN be memorised = {frac:.3f} "
            f"({s_on['steps_refused']}/{s_on['steps_checked']}), bar >= 0.05")
else:
    verdict("P58b", False, "no gate summary for the M-ON arm - the arm did not run")

# P58d - outcome, as worded and corrected
if s_on and m_off["last"]:
    print("  P58d FAIL AS WORDED: the prediction compared the PROBE loss of the gated arm against the probe "
          "loss of the un-gated arm, and the un-gated arm has none - with the gate off the trainer never "
          "computes a probe loss (the number does not exist). Corrected comparison, recorded beside it:")
    v_on, v_off = m_on["last"]["val_ppl"], m_off["last"]["val_ppl"]
    verdict("P58d", v_on <= v_off,
            f"held-out VAL ppl at the last epoch: gate ON {v_on:.2f} vs gate OFF {v_off:.2f} "
            f"(ratio {v_on / v_off:.3f}, bar <= 1.000); the probe is a 256-window subset of the same "
            f"held-out split, so the val number is the same kind of number on more windows")

# P58e - cost on the learnable corpus
if s_l:
    frac = s_l["refused_fraction"]
    verdict("P58e", frac <= 0.20,
            f"refused fraction on the corpus that CAN be learned = {frac:.3f} "
            f"({s_l['steps_refused']}/{s_l['steps_checked']}), bar <= 0.20 "
            f"[PREDICTED TO FAIL: the same rule measured 0.660 on the learnable synthetic corpus]")
else:
    verdict("P58e", False, "no gate summary for the L-ON arm")

# P58f - consistency
if s_on and s_on2:
    same = s_on["steps_refused"] == s_on2["steps_refused"]
    verdict("P58f", same, f"same seed, two runs: refused {s_on['steps_refused']} vs {s_on2['steps_refused']} "
                          f"of {s_on['steps_checked']} vs {s_on2['steps_checked']} steps")

# P58g - the per-step record
ok_g, txt = True, []
for arm in ("gate_m_on", "gate_m_on2", "gate_l_on", "gate_m_relgrad"):
    s = summary(arm)
    nrows = jsonl_rows(arm)
    good = bool(s) and nrows == s["steps_checked"] and nrows > 0
    ok_g &= good
    txt.append(f"{arm} {nrows} rows vs {s['steps_checked'] if s else '-'} steps")
verdict("P58g", ok_g, "one per-step JSONL row per training step: " + "; ".join(txt))

# P58h - direction
if m_off["last"] and m_on["last"]:
    t_on, t_off = m_on["last"]["train_loss"], m_off["last"]["train_loss"]
    verdict("P58h", t_on > t_off, f"final train loss: gate ON {t_on:.4f} vs gate OFF {t_off:.4f} "
                                  f"(the gate refuses the steps that fit the batch)")

# P58i - disjointness
ok_i, txt = True, []
for arm in ("gate_m_on", "gate_m_on2", "gate_l_on", "gate_m_relgrad"):
    s = summary(arm)
    o = s["probe_disjoint_overlap"] if s else None
    ok_i &= (o == 0)
    txt.append(f"{arm} overlap {o}")
verdict("P58i", ok_i, "probe windows also present in the train set (bar: 0): " + "; ".join(txt))

# P58j - relgrad + the gate, and lens/opt.py untouched
h = hashlib.md5(open("lens/opt.py", "rb").read()).hexdigest()
ok_j = bool(s_rg) and s_rg["steps_checked"] > 0 and ex.get("M-RELGRAD") == 0 \
    and h == "e1fec8a17dd460f1cc0e6f7bcd574423"
verdict("P58j", ok_j, f"relgrad + the gate ran {s_rg['steps_checked'] if s_rg else 0} steps, exit "
                      f"{ex.get('M-RELGRAD')}, lens/opt.py md5 {h} (recorded e1fec8a17dd460f1cc0e6f7bcd574423)")

print("\nNOT CLAIMED: no chat-accuracy claim of any kind; the numbers above are about the RULE firing and the")
print("held-out loss it protects, not about how well the model talks.")
