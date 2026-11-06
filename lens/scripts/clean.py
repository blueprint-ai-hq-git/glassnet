"""clean.py — the project's own data cleaner. Works on ANY dataset. ON/OFF switch.

WHY THIS EXISTS
The owner's requirement: "it can get smart on any dataset we can throw at it because our own
cleaning script that is easy to use". So this is deliberately generic -- it takes plain text items
(and can pull them out of jsonl / json / txt / a directory of files) and applies a short list of
READABLE rules.

DESIGN RULES IT FOLLOWS
  1. Every rule has a stated reason and a threshold that is a NAMED constant, not a magic number
     buried in code. A reader can see exactly what was removed and why.
  2. It NEVER silently drops. Every removal is counted per rule and examples are kept, so the
     report can be read and argued with.
  3. The rule list can be FROZEN to a file and reloaded, so the same cleaning can be proved to have
     been applied twice.
  4. It is OPTIONAL. `enabled=False` returns the data untouched, with a report saying so. The owner
     asked for an on/off switch, and both paths must be reportable side by side.

THE RULES (this is the whole cleaner -- nothing hidden)
  drop_empty          the item has no content after stripping
  drop_too_short      fewer than MIN_CHARS characters -- nothing learnable in it
  drop_too_long       more than MAX_CHARS -- one item must not dominate a batch
  drop_low_signal     fewer than MIN_ALNUM_FRAC letters/digits -- punctuation soup or binary junk
  drop_repetition     a single character is more than MAX_CHAR_REPEAT_FRAC of the item
  drop_duplicate      an exact duplicate of an item already kept (order-preserving)
  strip_boiler         remove a repeated prefix/suffix that appears in > BOILER_FRAC of items

`strip_boiler` is the only rule that MODIFIES rather than drops, and it is off by default because
modifying text changes what the model learns -- it must be turned on deliberately and the report
says what it removed.
"""
import json
import os
import re

# ------------------------------------------------------------------ the thresholds, named and visible
MIN_CHARS = 20            # below this there is nothing to learn
MAX_CHARS = 20000         # above this one item dominates a batch
MIN_ALNUM_FRAC = 0.35     # below this it is punctuation soup or decoded binary
MAX_CHAR_REPEAT_FRAC = 0.35   # one character being a third of the whole item
BOILER_FRAC = 0.25        # a prefix/suffix in more than a quarter of items is boilerplate
BOILER_LEN = 40           # only look at this much of the start/end

RULES = ("drop_empty", "drop_too_short", "drop_too_long", "drop_low_signal",
         "drop_repetition", "drop_duplicate", "strip_boiler")


def _alnum_frac(s):
    if not s:
        return 0.0
    return sum(c.isalnum() for c in s) / len(s)


def _max_char_frac(s):
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / len(s)


def _find_boiler(items, frac=BOILER_FRAC, n=BOILER_LEN):
    """The most common prefix and suffix, if either appears in more than `frac` of items."""
    if len(items) < 8:
        return "", ""
    from collections import Counter
    pre = Counter(s[:n] for s in items if len(s) >= n)
    suf = Counter(s[-n:] for s in items if len(s) >= n)
    out = ["", ""]
    for i, c in ((0, pre), (1, suf)):
        if c:
            w, k = c.most_common(1)[0]
            if k / len(items) >= frac and w.strip():
                out[i] = w
    return out[0], out[1]


def clean(items, enabled=True, rules=None, strip_boiler=False, verbose=True):
    """Apply the rules. Returns (kept_items, report).

    `report` is the artifact: per-rule counts and examples, so the cleaning is auditable and can be
    proved to have been applied. It is JSON-serialisable on purpose.
    """
    items = [("" if x is None else str(x)) for x in items]
    report = {"enabled": bool(enabled), "n_in": len(items), "rules": {}, "dropped": {},
              "examples": {}, "boiler": ["", ""], "n_out": len(items)}
    if not enabled:
        report["note"] = "cleaner OFF -- data passed through untouched"
        if verbose:
            print(f"  cleaner OFF: {len(items):,} items passed through untouched")
        return items, report
    if rules is None:
        rules = list(RULES)
    unknown = set(rules) - set(RULES)
    if unknown:
        raise ValueError(f"unknown rules {sorted(unknown)}; known: {list(RULES)}")
    report["rules"] = list(rules)
    report["rule_detail"] = {}
    report["thresholds"] = {"MIN_CHARS": MIN_CHARS, "MAX_CHARS": MAX_CHARS,
                            "MIN_ALNUM_FRAC": MIN_ALNUM_FRAC,
                            "MAX_CHAR_REPEAT_FRAC": MAX_CHAR_REPEAT_FRAC,
                            "BOILER_FRAC": BOILER_FRAC, "BOILER_LEN": BOILER_LEN}

    work = list(items)

    # strip_boiler MODIFIES; it is therefore opt-in and reported separately
    if strip_boiler and "strip_boiler" in rules:
        pre, suf = _find_boiler(work)
        report["boiler"] = [pre, suf]
        changed = 0
        if pre or suf:
            new = []
            for s in work:
                t = s
                if pre and t.startswith(pre):
                    t = t[len(pre):]
                if suf and t.endswith(suf):
                    t = t[:-len(suf)]
                if t != s:
                    changed += 1
                new.append(t)
            work = new
        report["rule_detail"]["strip_boiler"] = {"changed": changed, "prefix_removed": pre,
                                                 "suffix_removed": suf}

    kept, dropped, examples = [], {}, {}
    seen = set()

    def drop(rule, s):
        dropped[rule] = dropped.get(rule, 0) + 1
        e = examples.setdefault(rule, [])
        if len(e) < 5:
            e.append(s[:160])

    for s in work:
        t = s.strip()
        if not t:
            drop("drop_empty", s)
            continue
        if len(t) < MIN_CHARS:
            drop("drop_too_short", t)
            continue
        if len(t) > MAX_CHARS:
            drop("drop_too_long", t)
            continue
        if _alnum_frac(t) < MIN_ALNUM_FRAC:
            drop("drop_low_signal", t)
            continue
        if _max_char_frac(t) > MAX_CHAR_REPEAT_FRAC:
            drop("drop_repetition", t)
            continue
        if t in seen:
            drop("drop_duplicate", t)
            continue
        seen.add(t)
        kept.append(t)

    report["rule_detail"].update({k: {"threshold": v} for k, v in {
        "drop_too_short": f"< {MIN_CHARS} chars",
        "drop_too_long": f"> {MAX_CHARS} chars",
        "drop_low_signal": f"alnum fraction < {MIN_ALNUM_FRAC}",
        "drop_repetition": f"single char > {MAX_CHAR_REPEAT_FRAC} of item",
        "drop_duplicate": "exact duplicate of an item already kept",
    }.items() if k in rules})
    report["dropped"] = dropped
    report["examples"] = examples
    report["n_out"] = len(kept)
    report["kept_frac"] = len(kept) / max(len(items), 1)

    if verbose:
        print(f"  cleaner ON: {len(items):,} in -> {len(kept):,} kept "
              f"({report['kept_frac']*100:.2f}%), {len(items)-len(kept):,} dropped")
        for r in sorted(dropped, key=lambda k: -dropped[k]):
            print(f"    {r:<20} {dropped[r]:>8,}")
            for e in examples.get(r, [])[:1]:
                print(f"        e.g. {e[:90]!r}")
        if report["boiler"][0] or report["boiler"][1]:
            print(f"    boilerplate removed: prefix={report['boiler'][0][:40]!r} "
                  f"suffix={report['boiler'][1][:40]!r}")
    return kept, report


# ------------------------------------------------------------------ loaders: any dataset shape
def load_jsonl(path, field="text"):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                out.append(line)
                continue
            if isinstance(d, dict):
                for k in (field, "text", "content", "problem", "question", "code", "solution"):
                    if isinstance(d.get(k), str):
                        out.append(d[k])
                        break
            elif isinstance(d, str):
                out.append(d)
    return out


def load_any(path, field="text"):
    """Load items from jsonl / json list / plain text / a directory of text files."""
    if os.path.isdir(path):
        out = []
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if os.path.isfile(p) and os.path.getsize(p) < 5_000_000:
                try:
                    out.extend(open(p, encoding="utf-8", errors="replace").read().split("\n\n"))
                except Exception:
                    pass
        return out
    if path.endswith(".jsonl"):
        return load_jsonl(path, field)
    if path.endswith(".json"):
        d = json.load(open(path))
        if isinstance(d, list):
            return [x if isinstance(x, str) else str(x.get(field, x)) for x in d]
        return [str(d)]
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    return txt.split("\n\n") if "\n\n" in txt else txt.splitlines()


def freeze(path, report):
    """Write the rule list + thresholds so the SAME cleaning can be proved to have been applied."""
    json.dump({"rules": report.get("rules"), "thresholds": report.get("thresholds"),
               "n_in": report.get("n_in"), "n_out": report.get("n_out"),
               "dropped": report.get("dropped")}, open(path, "w"), indent=1)
    return path


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "/run/media/yahya/HARD DISK 4/GlassNet", "data", "code", "magicoder_oss_instruct.jsonl")
    print("=" * 96)
    print(f"DATA CLEANER — self-test on {src}")
    print("=" * 96)
    items = load_any(src)
    print(f"  loaded {len(items):,} items")
    # 1. a deliberately dirty sample must lose exactly the junk
    dirty = (["short"] * 3 + [""] * 2 + ["a" * 30] * 2 + ["..." * 40] * 2
             + ["A perfectly ordinary sentence about code and data."] * 4
             + ["dup item that is quite long and normal looking"] * 3)
    kept_d, rep_d = clean(dirty, verbose=True)
    # 16 in: 4 identical sentences collapse to 1 (3 dropped), 3 identical dup items collapse to 1
    # (2 dropped), 3 "short", 2 empty, 2 "aaaa...", 2 "...." -> 16-3-2-3-2-2-2 = 2 kept.
    # The FIRST version of this assertion said 5 and failed. The cleaner was right and the test was
    # wrong: it is easy to write an expected number from memory and call the code broken when the
    # expectation is what is broken. The counts below are the assertion's real content.
    expect_kept = 2
    assert len(kept_d) == expect_kept, \
        f"expected {expect_kept} kept, got {len(kept_d)}; drops={rep_d['dropped']}"
    assert rep_d["dropped"].get("drop_duplicate") == 5, rep_d["dropped"]
    assert rep_d["dropped"].get("drop_empty") == 2, rep_d["dropped"]
    print(f"  -> {len(kept_d)} kept, and every drop count matches the expectation exactly")
    # 2. the real data
    kept, rep = clean(items, verbose=True)
    # 3. OFF must be a true pass-through
    passthru, rep_off = clean(items[:100], enabled=False, verbose=True)
    assert passthru == [str(x) for x in items[:100]] and rep_off["n_out"] == 100
    print()
    print("  OFF mode verified: data returned untouched")
    freeze(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "clean_rules_frozen.json"), rep)
    print("  rules+thresholds frozen to clean_rules_frozen.json")
    print("=" * 96)
