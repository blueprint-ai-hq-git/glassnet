PUBLISH PLAN - community repo, prepared 2026-11-05 night. NOT PUBLISHED YET.

WHAT GOES IN (auditable, small, honest)
  lens/ (engine, model, ops, methods, scripts), tests, ARCH-OURS.md, LAYER-MAP.md, GATE-B-VERDICT.txt,
  ARCH-OURS-VERDICT.txt, DATA_LICENSES.md, README with the REAL numbers including the losses.

WHAT MUST NOT GO IN (and is now in .gitignore)
  lens/runs/ (124 GB of checkpoints), data/ (datasets, licence-bound), .venv*, any .env or key file.
  Repo size after exclusions should be a few MB, not gigabytes.

BEFORE THE FIRST PUSH - HARD CHECKS
  1. grep the whole tree for the API key and any password. Publishing a key is instant and permanent.
  2. Confirm every dataset in DATA_LICENSES.md, and confirm none is redistributed in the repo.
  3. Confirm LICENSE (MIT) is the one he wants, since he mentioned a licence-based income plan.
  4. README must state the real numbers: 37.37% best, 24.20% at 2M windows, architecture bet LOST,
     attention test LOST 7 runs, arithmetic 0.0% on unseen. A repo that oversells gets destroyed by
     its first reviewer.

HONEST CAVEAT ON THE MONEY PLAN
  Open source plus "million users paying licences" is not a plan by itself: MIT lets anyone use it free,
  including commercially. If income is the goal the choice has to be deliberate:
    MIT (adoption, no income)  |  Apache-2.0 + patent grant (adoption, no income)
    AGPL-3.0 (companies must open their changes - the only licence that creates licence income pressure)
    dual licence (open core AGPL + paid commercial)  <- the one that can actually pay
  That is a decision for him, not for me to assume.

DAILY LOOP FROM TOMORROW
  one experiment, its prediction written first, its real number recorded, then a commit with that number in
  the message. Old results are never deleted - they move to a results archive, so the repo shows the history.
