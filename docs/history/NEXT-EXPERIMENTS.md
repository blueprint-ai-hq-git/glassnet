# NEXT EXPERIMENTAL BLOCK (queued 2026-10-17, after the running chains)

All arms on the REPAIRED corpora (`*_honest`), so the trainer's own per-epoch val is honest too.
One change at a time; seed 0; 96 epochs unless stated; measured with the honest meter (RAW + HONEST
both printed).

H1  merged honest, d256, 96 ep            CONTROL: must reproduce S0's 9.84% within noise. If it does not,
                                          the repaired corpus changed something it should not have.
H2  merged honest, d256, lam_concept=0    No concept supervision: does forcing the layer to match named
                                          formulas cost accuracy? This isolates the READABILITY cost.
H3  merged honest, d256, window 64        More context per prediction.
H4  merged honest, d256, 192 ep           Does the schedule alone close the gap (the old records used 192)?
H5  merged honest, d512, 192 ep           Bigger + longer, still readable.

LATER, BIGGER QUESTION: a same-parameter-budget UNCONSTRAINED net (plain control, no concept layer) to
separate "the bottleneck costs accuracy" from "the data/schedule caps us". Needs a small code path; do
it after H2 answers the cheaper version of the question.
