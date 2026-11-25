# 2026-11-25 - decorrelation cancellation (FAILED its gate) and custom-arch audit (PARTIAL: P1/P2 never printed)

Source files, both rc=0, both launched by glassnet_chain.sh:
  lens/runs/chain_run-decorr-cancel-.out      (finished 16:48)
  lens/runs/chain_run-custom-arch-direct-.out (finished 16:40)

## 1. decorr-cancel: does a decorrelation penalty CONCENTRATE concept supply?

Question: the concept layer already failed to be load-bearing (supply spread thin). Does adding a decorrelation penalty
(lam_decorr) squeeze the margin into fewer, more readable concepts?

Pre-registered bar: top-8 concepts among the POSITIVE suppliers must carry >= 80% of positive supply (P1).

MEASURED:
  arm                  VAL ppl    VAL acc   concepts   top-8 of positive supply   P1
  ctrl  (lam 0.0)      953.87     7.31%     25/36      72.3%                      FAIL
  d02   (lam 0.2)      955.09     7.25%     25/36      70.7%                      FAIL

Supporting lines, same file:
  ctrl: 19 of 36 concepts supply POSITIVE margin; positive supply totals 236.042; top-8 ablated together = -6.3% of the
        all-concept effect; count-invariant concentration 81.6% (top half of positive units); window check
        max |sum(contribution) - margin| = 2.109e-15 (a check that CAN fail, and it passed).
  d02 : 17 of 36 positive; supply 62.767; top-8 ablated = -7.6% of the all-concept effect; concentration 76.7%;
        residual 2.220e-15.

VERDICT: DECORRELATION FAILS ITS GATE. It moved concentration the WRONG way (72.3% -> 70.7%) and cost ppl
(953.87 -> 955.09). Positive supply also collapsed 236.042 -> 62.767, i.e. the penalty spread the margin OUT rather than
concentrating it. This is consistent with every earlier concept measurement in this project: the concept layer is not
where the answer lives, and forcing it harder does not put it there.

## 2. custom-arch direct on/off: concept honesty audit, 8 concepts

MEASURED (per-concept "does it mean what its label says" audit, 4,000 windows each):
  direct_on  4 PASS / 4 FAIL   (FAIL: art 0.0000 fires, law, a_size 0.00025 fires, a_quantity)
  direct_off 5 PASS / 3 FAIL   (FAIL: art 0.0000 fires, a_size 0.00025 fires, a_quantity)

Two concepts fire essentially never (mentions_art 0.00000, mentions_a_size 0.00025) while scoring AUROC 0.52-0.59, which is
coin-flip. A concept that never fires cannot be carrying anything, whatever its AUROC says.

INSTRUMENT GAP, recorded honestly: the script's closing block printed only
  "compare the concept-share lines above (P1), the held-out accuracies (P2), the audit lines (P3)"
with NO concept-share lines and NO held-out accuracies anywhere in the file. So P1 and P2 are UNREADABLE from this run and
must not be inferred. Only P3 (the audit above) is a real result. This is the seventh instrument bug in the project's
history, and the fix belongs in the script before it is run again.
