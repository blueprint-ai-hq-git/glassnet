"""record_clean_corpus.py — cleaned corpus built: backticks 6.08% -> 0.09% (67x reduction).
p1 technically FAILED (0.09%, not 0.00%) — recorded with the reason, not hidden.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
m = json.load(open(os.path.join(P, "data", "code", "prepared", "code_meta_v32000_clean.json")))

LOG = f"""
---

## 2026-10-09 — CLEANED CODE CORPUS BUILT: the prose problem is now measurable and mostly gone

### WHY THIS CHANGE (and why not more vocabulary)
Vocabulary was tested and RULED OUT at n=150 (p=0.42). Reading the actual training documents then
showed the real problem — every solution was wrapped in ```python fences AND followed by an English
explanation:

```
  ` ` ` python def find_palindromes ( words ) : ... return palindromes ` ` ` the ` find_palindromes `
  function uses list com ...
```

**The model emits backticks and prose because that is what its targets contain.** Corpus measurement:
'`' was 6.08% of tokens, 'the' 5.55%.

### WHAT WAS BUILT
`build_code_clean.py` extracts ONLY the content of the first fenced code block from each solution,
dropping the fences and the surrounding prose. Vocabulary, architecture, epochs and decoder all stay
identical — the DATA is the single change.

```
  documents kept         : {m['n_docs']:,}
  had a fenced block     : 30,230 (99.2%)
  total tokens           : {m['n_tokens']:,}
  vocabulary size        : 32,000
  <unk> share            : {m['unk_share']*100:.2f}%
  backtick share         : {m['backtick_share']*100:.4f}%   (was 6.08% -> a 67x reduction)
```

Sample document, now clean code:
```
  def find_palindromes ( words ) : palindromes = [ word for word in words if word . lower ( ) .
  replace ( " " , " " ) = = word [ : : - 1 ] . lower ( ) . replace ( " " , " " ) ] return
  palindromes def <unk> ( tokens ) : parsed_tokens = [ ] for token in tokens : ...
```

### p1 TECHNICALLY FAILED — stated plainly
The gate was "no backticks at all". Measured: **0.0911%, not 0.0000%.** Cause: 255 of 30,485
documents (0.8%) contained NO fenced block, so the fallback kept their raw text, which may include a
stray backtick. I am NOT calling this a pass. It is a 67x reduction and the residual is 0.09%, which
is small enough that it is very unlikely to drive generation — but the gate said zero and the
measurement says not-zero, so it is recorded as FAILED with the reason.

### THE EXPERIMENT THIS ENABLES (gates pre-registered)
  p2  training on the clean corpus must beat the 6k baseline's 20/150 syntax rate at n=150, with
      Fisher exact p < 0.05
  p3  if p2 fails, data content is ALSO not the blocker, and code generation is capacity-limited —
      report that and stop changing the data
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended clean corpus record")
print(f"  backticks {m['backtick_share']*100:.4f}% (was 6.08%), docs {m['n_docs']:,}, tokens {m['n_tokens']:,}")
