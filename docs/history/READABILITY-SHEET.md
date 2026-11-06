# READABILITY SHEET — the 10 taught A3 concepts

**STEP 0 report for the owner. No training, no retuning, no renaming was done.**
**Model read:** `readable_K16_lr1.0_lc01_clusters16_seed0.npz` (the A3 winner: lr=1.0,
lam_concept=0.1, seed 0). One trained model, stated rather than hidden.
**Evidence:** `runs/t3/readability_evidence.json` · raw run: `runs/t3/readability.log`
**Firing rule:** the audit's own — a column fires when its activation is **> 0.5**. The
"firing rate" column is the fraction of the 10,124 validation windows above that.
**Examples:** the **5 highest-activation** windows for that column, decoded to words.
`<PAD>` is padding inside the 8-word window.

---

## The honest headline, first

**Most of the 10 taught concepts are word detectors, not ideas.** Counted strictly:

* **2 are phrase/construction detectors** — `[8] "for such a one as"` and `[9] "gave him"`.
  These are genuine multi-word patterns, and a human could name them.
* **2 are mixed clusters** where a meaningful pattern is present but diluted — `[0]`
  ("he" + "more … than") and `[6]` ("that upon"). Nameable, but not one clean idea.
* **6 are single-token or word-list detectors** — `[2] "And"`, `[7] "The"`,
  `[5] "you,"`, `[1] "them"`, `[3] "My/know"`, `[4] "they/If"`. A human would call these
  *"the word X"*, not a concept.

**No concept here is a conceptual category of the kind the original goal imagined**
("has loop", "pronouns"). The nearest thing to a category is `[4]`, which mixes the pronouns
`they/us` with the conditionals `If/let` — so it is not cleanly "pronouns" either.

That is the honest read, and it matters more than the 80% audit pass rate: **the audit
measures whether a column detects its pre-registered target, not whether the target was a
good idea.** A column that reliably detects the word "The" passes the audit, and it is still
just the word "The".

---

## The sheet

### [0] `he`, `more`, `As`, `hath`, `than` — firing rate 0.0776
1. `this <PAD> he is more <PAD> than <PAD>`
2. `<PAD> he is more <PAD> than <PAD> GRUMIO:`
3. `By this <PAD> he is more <PAD> than`
4. `he is more <PAD> than <PAD> GRUMIO: <PAD>`
5. `come, <PAD> he <PAD> <PAD> Why, sir, he`

**Human guess:** *"he" plus the comparative "more … than".* Target present in 5/5.
**Honest label: MIXED CLUSTER.** Two unrelated things — a pronoun and a comparison frame —
glued together. Fires on either.

### [1] `To`, `I'll`, `make`, `them` — firing rate 0.0634
1. `them both, These are their <PAD> bid them`
2. `are their <PAD> bid them use them well.`
3. `These are their <PAD> bid them use them`
4. `these <PAD> To my <PAD> and tell them`
5. `<PAD> To my <PAD> and tell them both,`

**Human guess:** *"them" with "To".* Target present in 5/5 — but 4 of 5 examples are driven
by `them`, not by `To`.
**Honest label: WORD LIST**, effectively a detector for `them`.

### [2] `And` — firing rate 0.0850
1. `<PAD> And <PAD> <PAD> for her <PAD> And`
2. `And he shall be <PAD> of <PAD> And`
3. `And yet I come not well. <PAD> And`
4. `And <PAD> <PAD> for her <PAD> And to`
5. `<PAD> Nay, then thou <PAD> it not; And`

**Human guess:** *the word "And".* Target present in 5/5.
**Honest label: SINGLE-TOKEN DETECTOR.** A perfectly good word detector. Not an idea.

### [3] `My`, `DUKE`, `You`, `know` — firing rate 0.0395
1. `thus <PAD> on <PAD> next you know My`
2. `My <PAD> <PAD> my <PAD> my <PAD> My`
3. `to <PAD> your <PAD> <PAD> <PAD> You know`
4. `<PAD> the <PAD> at the <PAD> You know`
5. `them know you one <PAD> <PAD> I know`

**Human guess:** *possessive "My" and the phrase "you know".* Target present in 5/5.
**Honest label: WORD LIST.** Two different things again (possessive + discourse phrase).

### [4] `they`, `let`, `If`, `us` — firing rate 0.0446
1. `<PAD> Yet, as they are, here are they`
2. `Yet, as they are, here are they come`
3. `<PAD> till they kiss their <PAD> Are they`
4. `have a present <PAD> If not, <PAD> they`
5. `Thy <PAD> why, <PAD> come, <PAD> let us`

**Human guess:** *pronouns "they/us" with conditionals "If/let".* Target present in 5/5.
**Honest label: MIXED CLUSTER.** The closest to a grammatical category in this set, but it
mixes pronouns with conditionals, so calling it "pronouns" would overstate it.

### [5] `KING`, `What`, `you,`, `RICHARD` — firing rate 0.0318
1. `<PAD> <PAD> <PAD> And, Will you, <PAD> you,`
2. `<PAD> God save you, sir! TRANIO: And you,`
3. `<PAD> God save you, <PAD> PETRUCHIO: And you,`
4. `you, <PAD> you, I will marry you. Now,`
5. `you, sir! TRANIO: And you, sir! you are`

**Human guess:** *the token `you,` (with the comma).* Target present in 5/5 — but every
example is driven by `you,`; `KING`, `RICHARD` and `What` never appear in its top firings.
**Honest label: SINGLE-TOKEN DETECTOR**, and a **misleading name**: the cluster is called
`KING_What_you,_RICHARD` because those words co-occurred at derivation time, but the column
only learned `you,`. This is exactly the naming hazard the audit was built to catch — and
here the audit passes it, because `you,` *is* in the pre-registered target set.

### [6] `That`, `upon`, `me,` — firing rate 0.0508
1. `That she shall <PAD> <PAD> an <PAD> That`
2. `<PAD> so well together, That upon <PAD> is`
3. `have <PAD> so well together, That upon <PAD>`
4. `That <PAD> thus upon my <PAD> Go, get`
5. `so well together, That upon <PAD> is the`

**Human guess:** *the construction "that upon".* Target present in 5/5.
**Honest label: PHRASE DETECTOR (weak).** It fires on `That` alone as often as on
`That upon`, so the phrase is real but not exclusive.

### [7] `The` — firing rate 0.0115
1. `the <PAD> <PAD> The <PAD> father, <PAD> The`
2. `<PAD> <PAD> The <PAD> father, <PAD> The <PAD>`
3. `<PAD> The <PAD> father, <PAD> The <PAD> <PAD>`
4. `The <PAD> father, <PAD> The <PAD> <PAD> <PAD>`
5. `LUCENTIO: All but the <PAD> HORTENSIO: The base`

**Human guess:** *the word "The".* Target present in 5/5.
**Honest label: SINGLE-TOKEN DETECTOR.** Also **the column that fails A3-1 on the firing
floor** (0.0115 < 0.02) despite AUROC 1.0000 — a perfect detector of a common word that
still cannot clear a 2% firing bar.

### [8] `For`, `one` — firing rate 0.0282
1. `For such a one as leaves a gentleman,`
2. `live in this <PAD> For such a one`
3. `<PAD> For such a one as leaves a`
4. `in this <PAD> For such a one as`
5. `this <PAD> For such a one as leaves`

**Human guess:** *the construction "for such a one as".* Target present in 5/5.
**Honest label: PHRASE DETECTOR — the best concept in the set.** Every top firing is the
same recurring four-word frame. A human could name this and it is not a word list.

### [9] `him` — firing rate 0.0187
1. `I gave him no <PAD> I gave him`
2. `him no <PAD> I gave him the <PAD>`
3. `gave him no <PAD> I gave him the`
4. `make him glad to seem <PAD> And give`
5. `him glad to seem <PAD> And give <PAD>`

**Human guess:** *the verb+pronoun frame "gave him".* Target present in 5/5.
**Honest label: PHRASE DETECTOR.** Fires on `gave him`/`give him`, not on `him` in general.
Also fails A3-1 on the firing floor (0.0187 < 0.02) with AUROC 1.0000.

---

## Summary table

| # | target words | human guess | honest label | target in top 5 | fires |
|---|---|---|---|---|---|
| 0 | he, more, As, hath, than | "he" + "more … than" | MIXED CLUSTER | 5/5 | 0.0776 |
| 1 | To, I'll, make, them | "them" | WORD LIST | 5/5 | 0.0634 |
| 2 | And | the word "And" | SINGLE TOKEN | 5/5 | 0.0850 |
| 3 | My, DUKE, You, know | "My" + "you know" | WORD LIST | 5/5 | 0.0395 |
| 4 | they, let, If, us | pronouns + conditionals | MIXED CLUSTER | 5/5 | 0.0446 |
| 5 | KING, What, you,, RICHARD | the token "you," | SINGLE TOKEN (misnamed) | 5/5 | 0.0318 |
| 6 | That, upon, me, | "that upon" | PHRASE (weak) | 5/5 | 0.0508 |
| 7 | The | the word "The" | SINGLE TOKEN | 5/5 | 0.0115 |
| 8 | For, one | "for such a one as" | **PHRASE (best)** | 5/5 | 0.0282 |
| 9 | him | "gave him" | PHRASE | 5/5 | 0.0187 |

**Strict count: 2 phrase detectors, 2 mixed clusters, 6 word/single-token detectors.**
Every column does fire on its target — the detection is real. The *concepts* are mostly not.

## What this implies for Target B (stated before running it)

Readable models here are being asked to win on perplexity and accuracy while carrying a
concept layer whose names are mostly single words. That is a real handicap and a real
explanation for the parity result in A3, and it is worth remembering if Target B comes out
as parity or behind: **the honest reading may be "the concept layer is not yet carrying
ideas worth carrying", not "readability is impossible."**
