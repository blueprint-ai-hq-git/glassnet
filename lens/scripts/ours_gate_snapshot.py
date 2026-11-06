"""ours_gate_snapshot.py — what OUR method's gates are actually doing, from the per-step logs it writes.

WHY: lens/scripts/run_ours_first.sh (queued 2026-10-23 15:46) predicted, BEFORE it ran, "THE GATE MUST ACT ...
If the run reports ~0% refused, the gate is not really acting and that is a finding against it, not a clean run."
That claim is decided by the per-step logs the trainer writes, not by the epoch line, so it is read here at any
moment while the run is in flight: lens/runs/chat/ours_d256_full_3_gates.jsonl (memorisation rule) and
..._agreement.jsonl (agreement rule). Read-only; writes one snapshot file (default OURS-METHOD-GATES-SO-FAR.txt).

Prints, per rule: how many decisions, how many REFUSED, the refusal rate, and the first/last 5 refusals with
their own numbers (train/probe losses and the gap for the memorisation rule; A_train/A_probe/ema for the
agreement rule) so a reader can see WHY a step was refused, not just that it was.
"""
import json
import os
import sys
import time

P = "/run/media/yahya/HARD DISK 4/GlassNet"
TAG = os.environ.get("OURS_TAG", "ours_d256_full_3")
OUT = os.environ.get("SNAP_OUT", os.path.join(P, "lens/runs/OURS-METHOD-GATES-SO-FAR.txt"))
CH = os.path.join(P, "lens/runs/chat")

rows = []


def log(s=""):
    print(s, flush=True)
    rows.append(str(s))


def load(fn):
    p = os.path.join(CH, fn)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


log("=" * 100)
log(f"OUR METHOD — WHAT THE GATES ARE DOING   snapshot {time.strftime('%F %H:%M')}   tag {TAG}")
log("source files: lens/runs/chat/%s_gates.jsonl (memorisation rule) and _agreement.jsonl (agreement rule)" % TAG)
log("prediction under test (lens/scripts/run_ours_first.sh, written before it ran): THE GATE MUST ACT -")
log("'if the run reports ~0% refused, the gate is not really acting and that is a finding against it'")
log("=" * 100)

for name, fn, keys in (
    ("MEMORISATION RULE (gap gate)", f"{TAG}_gates.jsonl",
     ("step", "epoch", "train_loss", "probe_loss", "ema_train", "ema_held", "gap")),
    ("AGREEMENT RULE (vs_probe)", f"{TAG}_agreement.jsonl",
     ("step", "epoch", "a_train", "a_probe", "ema_a", "batch_ce")),
):
    data = load(fn)
    log("")
    log(f"--- {name}   file {fn}")
    if not data:
        log("    NOTHING READ YET (no per-step line written) — no claim can be made from this file.")
        continue
    decided = [r for r in data if "accepted" in r]
    ref = [r for r in decided if not r["accepted"]]
    acc = [r for r in decided if r["accepted"]]
    log(f"    decisions logged {len(decided):>6}   accepted {len(acc):>6}   REFUSED {len(ref):>6}"
        f"   refusal rate {100.0*len(ref)/max(len(decided),1):5.1f}%")
    log(f"    last logged step {data[-1].get('step')}")
    if ref:
        log("    first 3 refusals, with their own numbers:")
        for r in ref[:3]:
            log("      " + json.dumps({k: r.get(k) for k in keys}))
        log("    last 2 refusals:")
        for r in ref[-2:]:
            log("      " + json.dumps({k: r.get(k) for k in keys}))
    else:
        log("    NO REFUSALS YET — read again later; the rule is not concluded until the run ends.")

log("")
log("=" * 100)
log("READ THIS AS: a non-zero refusal rate means the rule fired on real steps; a ZERO refusal rate at the end of")
log("the run means the gate did NOT act on this corpus at this scale, which is a finding AGAINST the gate.")
log("=" * 100)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print(f"written to {OUT}", flush=True)
