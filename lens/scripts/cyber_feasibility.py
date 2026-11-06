"""cyber_feasibility.py — can a LICENSE-CLEAN security capability be built from data already on disk?

WHY THIS SHAPE
  The cyber capability was blocked on "no licence-clean corpus offline". But the MIT-licensed code
  corpus (Magicoder-OSS / codeparrot-apps, 43k python docs) is already here, and SECURITY REVIEW is
  exactly the kind of task a transparent concept model is suited to: the answer must be auditable,
  and the reasons must be named ("this has a raw SQL string", "this calls eval", "this spawns a
  shell"). No downloads, no new licences, and the readable architecture is the point rather than a
  constraint.

THIS SCRIPT ONLY CHECKS VIABILITY (CPU, seconds):
  for a candidate set of transparent security concepts, how many documents actually carry each one?
  A concept with support near 0 or 1 cannot be learned or audited, so it must be dropped BEFORE any
  training, and the min_support bar is applied here, not after.

GATES (pre-registered)
  y1  at least 6 concepts survive min_support 0.02 (enough for a readable layer)
  y2  no single concept exceeds 0.95 support (a concept that is always true explains nothing)
  y3  at least one concept is REQUIRED to be rare (a security-relevant-but-uncommon pattern), so the
      task is not just "is this python"
"""
import os
import re
import sys
import collections

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import code_data                                          # noqa: E402

# Transparent, named, regex-measurable security concepts. Deliberately written as human-checkable
# rules: every one of these can be verified by eye against a document, which is the whole premise.
SEC_CONCEPTS = {
    "has_eval_exec":      r"\b(eval|exec)\s*\(",
    "has_shell_spawn":    r"\b(subprocess|os\.system|os\.popen|Popen|check_output|shell=True)\b",
    "has_raw_sql":        r"(SELECT|INSERT|UPDATE|DELETE)\s+.+\s+FROM|\.execute\s*\(\s*[\"']|cursor\.execute",
    "has_sql_param":      r"\?\s*,\s*\(|%s\s*,\s*\(|:\w+\s*,\s*\{|execute\s*\([^,]+,\s*\(",
    "has_input_read":     r"\binput\s*\(|sys\.argv|request\.(args|form|json|GET|POST)",
    "has_sanitize":       r"\b(escape|sanitize|quote|secure_filename|bleach|html\.escape|shlex\.quote)\b",
    "has_secret_literal": r"(password|passwd|secret|api_key|apikey|token)\s*=\s*[\"'][^\"']{4,}[\"']",
    "has_crypto":         r"\b(hashlib|hmac|bcrypt|sha256|md5|encrypt|decrypt|Fernet|cryptography)\b",
    "has_file_multi":     r"\b(open|Path)\s*\(|\.read\(\)|\.write\(",
    "has_pickle_load":    r"\bpickle\.(loads?|Unpickler)\b|yaml\.load\s*\(",
    "has_network":        r"\b(socket|requests\.|urllib|http\.client|urlopen|ftplib|smtplib)\b",
    "has_regex":          r"\bre\.(compile|match|search|sub|findall)\b",
}


def tokens_to_text(vocab, ids, lo, hi):
    return " ".join(vocab[i] for i in ids[lo:hi])


print("=" * 96)
print("CYBER FEASIBILITY — license-clean security concepts over the MIT-licensed code corpus")
print("=" * 96)
d = code_data.build()
vocab, ids, bounds = d["vocab"], d["ids"], d["bounds"]
n_docs = len(bounds) - 1
print(f"  corpus: {n_docs:,} documents, {len(ids):,} tokens (DOC_MODE={code_data.DOC_MODE})")

# rebuild raw text per doc (tokenised text is enough for these patterns; whitespace is normalised)
docs = []
for k in range(n_docs):
    lo, hi = bounds[k], bounds[k + 1]
    docs.append(tokens_to_text(vocab, ids, lo, hi))

counts = collections.Counter()
for name, pat in SEC_CONCEPTS.items():
    rx = re.compile(pat, re.I)
    hits = sum(1 for t in docs if rx.search(t))
    counts[name] = hits
    print(f"  {name:<20} {hits:>7,}  support {hits/n_docs:.4f}")

min_sup, max_sup = 0.02, 0.95
keep = [n for n, c in counts.items() if min_sup <= c / n_docs <= max_sup]
drop_low = [n for n, c in counts.items() if c / n_docs < min_sup]
drop_high = [n for n, c in counts.items() if c / n_docs > max_sup]
print(f"\n  keep (0.02 <= support <= 0.95): {len(keep)}")
for n in keep:
    print(f"    {n:<20} {counts[n]/n_docs:.4f}")
if drop_low:
    print(f"  dropped (too rare to learn/audit): {[(n, round(counts[n]/n_docs,4)) for n in drop_low]}")
if drop_high:
    print(f"  dropped (too common to explain):   {[(n, round(counts[n]/n_docs,4)) for n in drop_high]}")

rare = [n for n in keep if counts[n] / n_docs < 0.15]
y1 = len(keep) >= 6
y2 = not drop_high
y3 = len(rare) >= 1
print(f"\n  y1 >= 6 concepts survive            -> {'PASS' if y1 else 'FAIL'} ({len(keep)})")
print(f"  y2 no concept above 0.95 support    -> {'PASS' if y2 else 'FAIL'}")
print(f"  y3 at least one rare concept (<15%) -> {'PASS' if y3 else 'FAIL'} ({len(rare)})")
print(f"\n  VERDICT: {'VIABLE — a license-clean readable security classifier can be built from data on disk' if (y1 and y2 and y3) else 'NOT VIABLE as configured — adjust the concept set'}")
sys.exit(0 if (y1 and y2 and y3) else 1)
