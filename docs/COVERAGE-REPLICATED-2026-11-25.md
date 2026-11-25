# 2026-11-25 - the coverage price REPLICATED on 3 fresh seeds: 1,600 examples = 90.9% +/- 1.4

Source: lens/runs/COVERAGE-SEEDS.out, COVERAGE-SEEDS.json. Seeds 11, 12, 13 with independent data draws and independent
initialisation; one shared held-out set of 500 pairs per class so every seed is scored against the same target. 24,000 steps.

    seed   coverage 100   coverage 400   coverage 1600
      11        30.0%          56.8%           89.8%
      12        31.2%          50.2%           92.8%
      13        25.6%          51.4%           90.0%

    1,600 matched-distribution examples -> 90.9% mean, 1.4 sd, range 89.8-92.8%

Monotone on all three seeds. The pre-registered falsifier (any seed inverting 400 below 100) did not fire.

## Why this matters more than any single run today

Every other number today is a single seed. This one is a LAW with a spread attached:

**about 1,600 examples of a mechanism, drawn from the same distribution as the test, buy ~91% of it** (+/-1.4 points), at
hidden width 256, 24,000 steps. That is the first quantity in this project that a data budget can be built on, and it survived
the replication rule.

## The context that makes it usable

    coverage at matched distribution   the only reliable lever measured (91% for 1,600 examples)
    model width                         failed twice (0.017 on LM loss; -8.8 points at equal coverage)
    hints / trace slots / few-shot      failed entirely (+0.0 on unseen positions)
    curated boundary examples            HURT badly (-46.8 points, narrowed the training distribution)
    steps alone                         0.042 over 25,600 extra steps
    forgetting                          SOLVED by replay (93.5 points lost -> 2.5, buffer 1,024 clears the bar)

So the work is not cleverness and not parameters. It is generating many verified examples that match the distribution the model
will be tested on.
