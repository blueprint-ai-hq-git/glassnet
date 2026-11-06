# SHUTDOWN NOTE - 2026-10-15 night

## Machine state
Nothing of ours is running. GPU idle (0% util, 229 MiB, 15.4 W). No trainers, no matrix, no supervisors,
no watch loop. Safe to shut down. (Hermes cannot issue a shutdown itself - use RustDesk ID 247022723.)

## Saved tonight (all inside the GlassNet folder)
* `research/open-model-landscape.md` (17.7 KB) - community/open-training landscape: who runs what,
  community contribution model, licenses, whether logs are released.
* `research/tiny-auditable-ml-market-assessment.md` (13.6 KB) - market assessment: who buys what,
  observability vendors vs readable models, 5 positioning options A-E with evidence.
* `research/task-0..3.log` - raw transcripts of the four research tracks (so nothing is lost even if
  the two still-running tracks die with the machine).
* `LOG.md`, `BUGS.md` - today's diagnosis session: two harness bugs, three invalidated numbers, the
  corpus A/B that exonerated the merged corpus.

## Tomorrow, in order
1. Re-measure cleanly: FRESH d512 merged run (no resume), watched epoch by epoch, against the resumed
   M0 that collapsed. The resume is the only remaining difference - that is the open question.
2. Re-run the two research tracks that were still in flight (cheap, ~4 min each): concept-bottleneck /
   interpretable-LM prior art, and interpretability-evaluation + licensing-provenance standards.
3. Restrict `matrix_summary.py` to the merged vocabulary family (prepared_ext / chat / big share
   md5 01b1eba6f8) - it currently compares against the 63.85% reference, which uses a DIFFERENT
   vocabulary (82e0edd0). That comparison is invalid and must be impossible to print.
4. Fix the logger: append `history`, never overwrite (`_log.json` overwrite destroyed M0's ep1-29).
5. Build ONE self-contained publish package for external review (architecture + trainer + audit
   harness + all numbers + licenses), then decide the release.

## Open decision for Yahya (license)
Recommended, adoption-first: code Apache-2.0 (explicit patent grant + attribution NOTICE), docs and
data CC BY 4.0, weights Apache-2.0 with a model card that leads with the limits, plus a DATED public
release (Zenodo DOI / arXiv preprint) so the method is prior art and nobody can patent it against us.
Protection-first alternative: AGPL-3.0 for the code - any modified version, including one served over
a network, must stay open. Costs adoption by companies that will not touch copyleft.
Cannot be guaranteed by any license: stopping an independent lab from training something similar.
What a license CAN do is require attribution, keep derivatives open (copyleft), and - with a dated
publication - block anyone from claiming our specific method as their own invention.
