# GLASSNET — THE GOAL (owner's words, 2026-10-05)

**Status: GOAL DEFINITION. Nothing is running. This file is the target; the plan below it is
proposed, not started.**

---

## 1. What the owner said, restated in one place

> "We are like 80% able to read it properly. Set the goal to make it 100% fully understand
> it, and it carries over on big models too, and it works even on smaller models — for the
> smaller models to get even much smarter, even the bigger ones too. Make it fully
> understandable so we can fully read it. And invent our own training method: one will be
> like distillation — we can take a model, make data, put it into another model, and it can
> learn and get much smarter than the model it was from, completely from scratch. Even make
> a method for fine-tuning, and also training it from complete scratch with our own datasets."

**Broken into four goals:**

* **G1 — 100% readable.** Everything the network does is understandable to a human. Not 80%.
* **G2 — it carries over across sizes.** The same method works on **small** models (and makes
  them **smarter**) and on **big** models.
* **G3 — our own training method: self-improvement by distillation.** Take a model, have it
  produce data, feed that to another model, and the second one ends up **smarter than the
  one it came from** — from scratch, no borrowed weights.
* **G4 — our own fine-tuning method, and our own from-scratch training method** on our own
  datasets.

## 2. What "100% readable" has to mean, or it means nothing

"80%" today is not a guess — it is **8 of 10 concepts passing the audit** (Target A3). But
the audit only checks *"does this neuron detect the thing its name claims?"* It does **not**
check that the name is a good idea. So 100% must be defined so it cannot be faked:

**A concept counts as READABLE only if all four hold:**

| # | requirement | how it is measured |
|---|---|---|
| R1 | **it passes the audit** | Meter-03: AUROC ≥ 0.55, contribution ≥ 0.25, fires ≥ 2%, against its own pre-registered target |
| R2 | **its name is honest** | **name precision**: no word in the name that the neuron does not actually use (closes the `KING_What_you,_RICHARD` → `you,` hole) |
| R3 | **a human can say what it is** | the name is an idea a person would recognise, not a word list |
| R4 | **the name predicts the firing** | **an independent reader, given only the name, can pick out which windows it fires on** — this is the real test of "understandable", and it is the one that cannot be gamed |

**G1 is met at 100% when every concept column satisfies R1–R4**, and the score is reported as
a count, not a vibe.

**Honest note on R4:** it needs an independent reader. The other AI is a legitimate reader for
it, and so is the owner. This is the only metric in the project that is judged by a human
rather than computed, and that is deliberate — "understandable" is a human property.

## 3. What "carries over to big models" can honestly mean here

**The catch, stated plainly and up front:**

* Tthe development machine is a **a 6 GB consumer GPU with 6 GB** and no tensor cores. The owner's own LIMIT
  RULE says: climb as far as tthe development machine goes, then **stop and report where and why**.
* So **G2 can be *built* size-agnostic and *tested* up to roughly 20M parameters here.**
  A claim that it works on genuinely big models (billions of parameters) **cannot be measured
  on this hardware** and would be a **GUESS**, labelled as one — never presented as a result.
* What *can* be shown honestly: the readability metrics **hold or degrade gracefully** as size
  grows across the rungs this GPU can reach, with the stopping point reported as a result.

**What can be shown about "small models get smarter":** this one is fully testable here, and
it is the interesting one — a small model using our method versus the same small model without
it, same task, same budget. **Note the honest tension: A3 measured readable as *slightly
worse* than the plain normal on both accuracy and perplexity.** So "smarter" is not
established — it is exactly what the work has to prove.

## 4. The training methods (G3, G4)

* **G3 — self-improvement by distillation.** A model generates data; a second model trains on
  it; the second should end up **better than the first**. No borrowed weights: the teacher is
  **our own model**, which keeps the from-scratch rule intact.
* **G4a — our own fine-tuning method.** A small, readable, own-code procedure for adapting a
  model to new data.
* **G4b — our own from-scratch training method** on our own datasets, with the licenses
  recorded in `DATA_LICENSES.md`.

**One thing to check before building G3, because the owner's rules require it** ("asks if
findings are NEW/known; lit-check BEFORE building"): self-training / self-distillation as a
*family* is well known (a model generates data, a model trains on it). The literature check
must be done **before** claiming anything is new. **What may genuinely be new is doing it
through a readable concept layer**, where the student's concepts can be inspected — that is
the claim worth testing, and it has to be tested against the known methods, not in isolation.

## 5. Proposed ladder (each rung gated, one change at a time)

| rung | goal | gate |
|---|---|---|
| **1** | finish the readability definition: R2 (name precision) and R4 (reader test) built and pre-registered | both metrics exist, run on the A3 model, old vs new reported |
| **2** | **real concepts** — speaker names, pronouns, questions instead of single words | the R1–R4 count rises well above today's 8/10, and readability does not cost accuracy/perplexity |
| **3** | **G1 at 100%** on the current task | every column passes R1–R4, reported as a count |
| **4** | **G2 small-vs-big** — the same method across sizes this GPU can reach | readability holds as size grows; the stopping point is reported as a result |
| **5** | **G4b** — own from-scratch training method on own datasets | documented, reproducible, licensed |
| **6** | **G4a** — own fine-tuning method | reproducible; no borrowed weights |
| **7** | **G3** — self-improvement distillation, student beats teacher | student > teacher on the held-out metric, 3 seeds, after a literature check |

**Rungs 1 and 2 are already half-built:** they are exactly the two items already sitting in
`QUEUE-AFTER-B.md` (the name-precision check, and meaningful targets). The owner's new goal
and his queued items turn out to be the same road.

## 6. What stays true

* Work only inside the GlassNet folder. Delete nothing. Keep every failed run.
* **Never change a bar after seeing results.** No "passed in spirit". Parity is never a win.
* Validation for decisions; the test set at most once per milestone, logged.
* From scratch: own tensors, own forward/backward, own losses, own optimiser. No pretrained
  weights, no library autograd.
* Safe licenses only, recorded in `DATA_LICENSES.md`. **No license is chosen for GlassNet.**
* Label anything beyond what tthe development machine can measure as a **GUESS**.
* One change at a time, prediction written before the run.

## 7. What needs the owner

**One thing only: is this the goal, stated correctly?** Specifically:

1. Is **R4 (an independent reader must be able to predict the firing from the name alone)** the
   right definition of "fully understandable"? It is the strongest honest definition I can
   build, and it is human-judged by design.
2. Given the 6 GB limit, is it acceptable that **"works on big models" is built-for but only
   *tested* up to ~20M parameters here**, with anything beyond labelled a guess?
3. Which rung first — or do you want rungs 1 and 2 done together, since they are the same
   road?
