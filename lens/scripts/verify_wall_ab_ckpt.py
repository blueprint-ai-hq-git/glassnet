#!/usr/bin/env python3
"""Verify the wall A/B arm checkpoints against the A/B log. READ-ONLY.

For every arm that has a "final VAL" line in lens/runs/WALL-AB.txt:
  * which epoch the log named NEW BEST (that is the file best.npz must hold),
  * sha256(best.npz) vs sha256(ep<N>.npz) and array_equal over every model array,
  * the arm identity recorded inside the checkpoint (__meta__: seed, direct,
    direct_mode, direct_rank),
  * the parameter count counted off the stored arrays, against the count the
    trainer announced in the log.

Nothing is written and no checkpoint is touched; the point is to prove that an
arm's number and its checkpoint belong together (arm 1 lost that link earlier
today: docs/history/WALL-AB-ARMS.md, "Provenance defect").

Usage: .venv-gpu/bin/python3 lens/scripts/verify_wall_ab_ckpt.py [log]
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lens/runs/WALL-AB.txt")
CKPT = os.path.join(ROOT, "lens/runs/chat")

# Markers can carry a timestamp prefix (run_wall_rep.sh publishes them through say(), so "-----" is not at
# column 0: "2026-11-06 18:31:30 ----- off seed 2 attempt 1/2 : ..."). The prefix is optional here, and the
# runner's failure lines ("-----off seed 2 GAVE UP ...", "----- SKIP ...") must not open a block.
RE_START = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?-----\s*(off|on|r64)\s+seed\s+(\d+)\b")
SKIPWORDS = ("GAVE UP", "SKIP", "INCOMPLETE", "DONE")
RE_FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+)/(\d+)")
RE_BEST = re.compile(r"NEW BEST held-out ([\d.]+)% at epoch (\d+) .*?-> saved (\S+_best\.npz)")
RE_EP = re.compile(r"^\s+ep (\d+)/3 .*VAL ppl ([\d.]+) acc ([\d.]+)%")
RE_PARAMS = re.compile(r"model: ([\d,]+) params")
# the block's own "saved to .../chat/<tag>_*" line names the tag, so a block whose tags are not
# wall_<arm>_s<seed> (e.g. wall_rep_off_s2) can be verified without a flag and without guessing.
# NOTE: the project path contains spaces ("HARD DISK 4"), so this must be .* and not \S* - measured, the
# first version silently fell back to wall_<arm>_s<seed> and "verified" four checkpoints that do not exist.
RE_SAVED = re.compile(r"saved to .*/chat/(\S+?)_\*")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blocks(lines):
    """Split the log into (arm, seed, body) runs; SKIP/failure markers are not runs.

    A log whose FIRST run predates the marker convention has no "----- off seed N" line at the top (the seed-0
    200k run in DATASCALE-HONEST.txt is one). Without this, that run is invisible and the tool reports only
    whatever was appended after it - which is how a stale 41-concept block got to look like the published 17.43%
    arm. So: if the first marker is not the first content line, emit the leading text as its own arm, labelled
    UNMARKED, and let the save line inside it supply the tag.
    """
    first_marker = None
    for i, line in enumerate(lines):
        if RE_START.match(line) and not any(w in line for w in SKIPWORDS):
            first_marker = i
            break
    if first_marker is None or first_marker > 0:
        head = lines[:first_marker] if first_marker is not None else lines
        if any("METHOD:" in l or "final VAL" in l for l in head):
            yield ["UNMARKED", -1, head]
    lines = lines[first_marker:] if first_marker is not None else []
    cur = None
    for line in lines:
        m = RE_START.match(line)
        if m and not any(w in line for w in SKIPWORDS):
            if cur:
                yield cur
            cur = [m.group(1), int(m.group(2)), [line]]
            continue
        if line.startswith("-----") and cur:
            yield cur
            cur = None
            continue
        if cur is not None:
            cur[2].append(line)
    if cur:
        yield cur


def main():
    with open(LOG, "r", errors="replace") as f:
        lines = f.readlines()

    rows = []
    for arm, seed, body in blocks(lines):
        final = None
        best = None
        eps = {}
        params_announced = None
        saved_tag = None
        for line in body:
            m = RE_FINAL.search(line)
            if m:
                final = (float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)))
                continue
            m = RE_BEST.search(line)
            if m:
                best = (float(m.group(1)), int(m.group(2)), m.group(3))
                continue
            m = RE_EP.match(line)
            if m:
                eps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                continue
            m = RE_PARAMS.search(line)
            if m and params_announced is None:
                params_announced = int(m.group(1).replace(",", ""))
                continue
            m = RE_SAVED.search(line)
            if m:
                saved_tag = m.group(1)
        if final is None:
            continue  # aborted/restarted block, no arm number to verify
        tag = saved_tag or ("wall_%s_s%d" % (arm, seed))
        rows.append(dict(arm=arm, seed=seed, tag=tag, final=final, best=best, eps=eps,
                         params_announced=params_announced))

    print("LOG %s" % LOG)
    print("%-14s %-9s %-7s %-7s %-6s %-12s %-9s %-7s %s" % (
        "arm", "final", "ppl", "concepts", "params", "best=ep?", "sha ok", "arrays", "meta"))
    bad = 0
    for r in rows:
        tag = r["tag"]
        ep3 = os.path.join(CKPT, tag + "_ep3.npz")
        bestp = os.path.join(CKPT, tag + "_best.npz")
        note = []
        meta_ok = "?"
        arrays_ok = "?"
        sha_ok = "?"
        best_epoch = r["best"][1] if r["best"] else None
        if r["best"]:
            # the saved filename in the log must be this arm's own
            if os.path.basename(r["best"][2]) != tag + "_best.npz":
                note.append("FOREIGN-BEST-NAME")
        try:
            z3 = np.load(ep3, allow_pickle=False)
            zb = np.load(bestp, allow_pickle=False)
            m3 = json.loads(str(z3["__meta__"]))
            mb = json.loads(str(zb["__meta__"]))
            KNOWN = {"off": ("off", False), "on": ("tied", True), "r64": ("lowrank", True)}
            if r["arm"] in KNOWN:
                exp = KNOWN[r["arm"]]
                meta_ok = (m3["direct_mode"] == exp[0] and bool(m3["direct"]) == exp[1]
                           and m3["seed"] == r["seed"])
            else:
                # not a wall-A/B arm (datascale / nogate / sgd / UNMARKED): there is no pinned expectation for
                # the direct head, so report what the checkpoint says instead of inventing a verdict. The seed
                # is only comparable when the block carried a marker.
                dm, dd = m3.get("direct_mode"), bool(m3.get("direct"))
                meta_ok = ("seed ok" if r["seed"] >= 0 and m3["seed"] == r["seed"]
                           else ("seed n/a" if r["seed"] < 0 else "SEED-DIFF"))
                note.append("direct=%s/%s" % (dm, dd))
            params = int(sum(z3[k].size for k in z3.files if k != "__meta__"))
            if r["params_announced"] and params != r["params_announced"]:
                note.append("PARAMS-DIFF %d vs log %d" % (params, r["params_announced"]))
            if best_epoch is not None:
                epf = os.path.join(CKPT, "%s_ep%d.npz" % (tag, best_epoch))
                sha_ok = sha256(epf) == sha256(bestp)
                ze = np.load(epf, allow_pickle=False)
                arrays_ok = all(np.array_equal(ze[k], zb[k]) for k in ze.files) and \
                    sorted(ze.files) == sorted(zb.files)
                note.append("ep%d" % best_epoch)
        except Exception as e:  # noqa: BLE001 - report, never hide
            note.append("ERROR %s: %s" % (type(e).__name__, e))
            params = -1
        # meta_ok is True for the wall A/B arms (pinned expectation) and a short string for arms that have no
        # pinned expectation ("seed ok" / "seed n/a"). Both are acceptable; SEED-DIFF is not. Counting the
        # string case as a failure made a perfectly verified checkpoint (17.43%, sha and arrays identical) print
        # as a provenance problem, which is the opposite of this tool's job.
        meta_pass = meta_ok is True or (isinstance(meta_ok, str) and meta_ok in ("seed ok", "seed n/a"))
        ok = meta_pass and arrays_ok is True and sha_ok is True and not any(
            n.startswith(("FOREIGN", "PARAMS-DIFF", "ERROR")) for n in note)
        bad += 0 if ok else 1
        print("%-14s %-9s %-7.2f %-7s %-6d %-12s %-9s %-7s %s %s" % (
            tag,
            "%.2f%%" % r["final"][1],
            r["final"][0],
            "%d/%d" % (r["final"][2], r["final"][3]),
            params,
            str(best_epoch),
            str(sha_ok),
            str(arrays_ok),
            str(meta_ok),
            " ".join(n for n in note if n)))

    print("")
    print("arms verified: %d, arms with a provenance problem: %d" % (len(rows), bad))
    print("NOTE: a number is only quoted in docs from the log line; this script proves the")
    print("      checkpoint that carries it is the one the log named, not a restarted twin.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
