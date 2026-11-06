# CONCEPT TARGETS 02 — WORD CLUSTERS (supervised concept loss, v2)

**PRE-REGISTERED: 2026-10-04, before computing the clusters or running anything.**
**Authority:** owner condition 4 — "try richer concept targets (word clusters, not
single words) and keep the supervised/unsupervised split."
**Meter:** Meter-02 (`METER-RULE-02.md`): AUROC ≥ 0.55 AND contribution ≥ 0.25 AND
firing rate ≥ 2% of validation windows. Bars unchanged.

This file freezes the METHOD before any cluster exists. The cluster contents are data
produced by this method; the method itself may not be edited after seeing results.

---

## 1. Why clusters

Single-word targets are thin: 8 of the 16 sat below the 2% firing floor, and a concept
trained on `What` is a detector of one token. A cluster target asks a concept to detect
a *family* of related words, which is a more concept-like job, and its support is the
union of its members' supports (so it is far more likely to clear the 2% floor
legitimately).

## 2. The method (FROZEN)

1. **Source: TRAIN split only** (`d["Xtr"]`, the same 8-token windows the model trains
   on). Val and test text are never read by the derivation. No external corpus, no
   pretrained vectors, nothing downloaded.
2. **Candidate words:** vocabulary entries that are not in the STOP list, length ≥ 2,
   with TRAIN document frequency ≥ **1% of windows**. Take the top **100** by document
   frequency (ties broken by vocabulary index).
3. **Co-occurrence:** for each candidate pair (i, j), count windows containing **both**
   words (symmetric; diagonal = own document frequency).
4. **Transform:** PPMI (positive pointwise mutual information) on those counts, with
   `pmi = log( p(i,j) / (p(i)p(j)) )`, negatives clipped to 0. The matrix is
   L1-normalised per row before clustering.
5. **Clustering:** k-means, **k = 16**, with a **fixed seed (0)**, 100 iterations,
   k-means++ initialisation, implemented with numpy only (no sklearn).
6. **Targets:** cluster c's target for a window = 1 iff the window contains **at least
   one** member word of c. Concept column c (0..15) is supervised on cluster c.
7. **Class balancing:** identical to v1 — positive terms weighted by
   `pos_weight = (1 - p) / p`, where `p` is that cluster's TRAIN base rate.
8. **Columns 16..K-1 are NOT supervised** and remain the circularity control.
9. Clusters are ordered by TRAIN support (descending) and assigned to columns 0..15, so
   the column index is deterministic.

## 3. What is reported (fixed in advance)

* The cluster membership list and each cluster's TRAIN support (recorded here after
  computation, by this method).
* **The number of clusters clearing the 2% support level** — the honest measure of
  whether clusters fixed the thin-target problem.
* The full Meter-02 audit with the **supervised vs unsupervised split**, at floors
  1% / 2% / 5%.
* Accuracy/perplexity against the matched normal, same margin clause as before.
* The lie probe result, labelled with which clause caught it.

## 4. Predictions (written before running)

1. **Support:** cluster targets are unions, so **at least 12 of 16** clusters will have
   TRAIN support ≥ 2%, against 8 of 16 for single words. If this fails, clusters did
   not fix the thin-target problem and I will say so.
2. **Supervised columns** will pass at ≥ 90% (they are trained on the target the audit
   asks about — circular, and reported as such).
3. **Unsupervised columns** are the honest number. Baseline is 29.2% at Meter-02
   (unchanged by the floor). Prediction: clusters move it **little** — I expect
   **25–40%**, because the earlier supervised run's apparent unsupervised gain
   (30.2% → 50.0% raw) was itself partly floor-inflated and fell to 47.9%.
4. **The 50% bar will NOT be met on the unsupervised half.** If it is, that is a real
   result and I will say so plainly rather than hedge.
5. **Perplexity:** clusters are broader targets, so the auxiliary term should interfere
   less with the task than v1's thin targets. Prediction: readable ppl within 1.0 of
   the matched normal (v1 was within 5.5).

## 5. What is NOT changed

* The audit bars, the selection rule, the test-set discipline (validation only, no test
  touch), the supervised/unsupervised split reporting.
* The circularity caveat stands and is reported with every number: the namer derives a
  name FROM a column's own firings, so supervised columns pass partly by construction.
  An audit-design change (auditing supervised columns against their PRE-REGISTERED
  cluster target instead of a derived name) is proposed separately and is **not** part
  of this run without owner approval.

## 6. Budget

Same as v1: `lam_concept ∈ {0.1, 0.5, 1.0}` × lr ∈ {0.5, 1.0} for tuning (1 seed,
validation perplexity, frozen `SELECTION-RULE.md`), then 3 seeds at the selected lr,
both models, then the Meter-02 audit. Validation only.
