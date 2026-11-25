# 2026-11-25 - CAPACITY IS NOT THE CONSTRAINT: 3.5x the parameters bought nothing

Source: lens/runs/CAPACITY.out (exit 0, fixed script), lens/runs/CAPACITY_d128.json, lens/runs/CAPACITY_d256.json.

The earlier capacity run was VOID (batch_np scored training windows as held-out). This is the fixed rerun.

    arm                             params      steps 25,600   steps 51,200
    d=128 hidden=512            5,476,816          4.20204        4.12436
    d=256 hidden=1024          19,340,240          4.18339        4.14131

Identical pool (200,000 distinct windows), step budget, seed, learning rate and held-out set (256 windows never trained).

    P1 wide reaches lower held-out        NOT MET at the endpoint (4.14131 vs 4.12436, narrow is 0.017 BETTER)
    P2 gain at least 0.05                 NOT MET (the wide arm is 0.017 worse)
    FALSIFIER (<0.02)                     FIRED

Both arms sit within 0.02 of each other, which is the same size as the GPU run-to-run spread measured today (0.011-0.013)
from cupy's order-nondeterministic scatter-add. So the honest statement is that width made NO measurable difference.

## The three levers, ranked by what they actually bought

    distinct data   20k -> 200k pool moved held-out by 3.445   <- the only real lever
    step budget     25,600 -> 51,200 moved it by 0.042
    model width     5.5M -> 19.3M params moved it by -0.017 (nothing)

This closes the model-size avenue at this scale and points every remaining hour at generating more DISTINCT, VERIFIABLE
data. That is a resource question, not an architecture question, which is why the self-verified maths/code generator is the
next build rather than another width sweep.
