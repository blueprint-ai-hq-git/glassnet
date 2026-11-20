

---

# PRIVACY FIX: SIX "personal name" EXEMPTIONS HID A REAL LEAK (2026-11-20 14:52)

Found while pushing the anatomy scan, and it is more important than the scan.

## What happened

`tools/check_pii.py` has an exact-match allow list (`tools/pii_allow.txt`) that suppresses hits by `(rule, filename, sha1(line))`.
Six of its entries were **`personal name`** exemptions. Their effect: the owner's personal name appeared in **five public files** -
`STATUS.md`, `docs/BENCH-V0.md`, `docs/RL-V1-METHOD.md`, `docs/MEM-SPLIT.md`, `docs/READABLE-PATH-EXPLAINS-7PCT.md` - and the gate
reported **"HITS: none"** for all of them, forever.

That is a worse failure than a missed pattern: the checker was not blind, it had been told to ignore the exact lines.

## Proof it was real, not theoretical

Removing only the `personal name|...` lines from the allow list (backup kept: `tools/pii_allow.txt.bak-20261120`) and re-running the
gate produced, immediately:

```
HITS
     HIGH personal name    line 4     <name>
     HIGH personal name    line 53    <name>
     HIGH personal name    line 844   <name>
     HIGH personal name    line 34    <name>
     HIGH personal name    line 3     <name>
     HIGH personal name    line 46    <name>
```

Six HIGH hits across the five files, all previously suppressed.

## The fix

- Allow list: 6 `personal name` exemptions **removed** (4 entries remain: the intentional `LEAK.md` records of a mounted disk path,
  disk label, machine GPU and real email, which exist on purpose so the leak is documented rather than hidden).
- Text: the five public files rewritten to say "the project owner" instead of the name. Public copies only; internal copies unchanged.
- Gate re-run: **HITS: none**, and `grep -ri` for the name across the whole public tree returns **nothing**.

## The rule this establishes

**An allow list may only exempt things that are allowed to be published, never a category that is banned.** `LEAK.md` documenting a
leaked disk label is legitimate. A `personal name` exemption is self-defeating: it exists to make a build pass, and it converts a
blocking check into a silent one. If a public doc needs to say who asked for something, it says "the project owner".

This is the third instrument defect of the same family found today (a summary that printed "every arm failed" over three passing
arms, a concentration bar that was vacuous at 9 units, and now a privacy gate exempting its own highest-severity rule). All three had
the same shape: **the tool reported success while the truth was the opposite.**
