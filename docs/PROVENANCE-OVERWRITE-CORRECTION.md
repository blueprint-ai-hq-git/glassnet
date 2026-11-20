

---

# CORRECTION: AN UNGUARDED BOOT RE-RUN OVERWROTE THE CHECKPOINT BEHIND THE MEMORY-LINE RESULT - AND THREAD COUNT CHANGES RESULTS (2026-11-20 09:20)

The memory-line closure published on 2026-11-19 (age key end-to-end: trainer **+0.14**, independent **-0.06**) cites
`lens/runs/chat/win6kage2_K8_ep6000.npz` as its source file. **Those bytes no longer exist.** On the 08:53 boot,
`lens/scripts/run_after_boot.sh` line 38 re-ran the entire age-key arm unconditionally - there was no "already done" guard - and retrained
`win6kage2` from scratch over the top of it. The file's mtime is now **2026-11-20 09:13**, size 7,404,494, sha256
`48abda39e04092d722ad1cbbe8469996dcb56b95531dec1d87215d8ec3972399` (recorded in `lens/runs/PROVENANCE-win6kage2-2026-11-20.txt`).

## The guard, and why it is a marker AND a checkpoint

Re-running a finished experiment silently is worse than a crash, because the crash is visible. The boot script now requires **both** the checkpoint
**and** `lens/runs/AGEKEY-E2E-DONE` to be absent before it will run the arm, it is pinned to two cores (`taskset -c 8,9`, nice 19, ionice c3), and
the skip decision is printed:

```
age-key arm already complete (checkpoint + marker present) - NOT re-running it
```

Verified: `SKIP-NOT-RE-RUN`.

## The unpinned re-run also stole the live experiment's CPU

The re-run was launched unpinned across all 12 cores, while `boost50` is pinned to cores 6,7. The cost, straight from the resume log's own timings:

```
steps 2400 -> 2825:  per-25-step wall time went 17s -> 60, 75, 84, 80, 79, 80, 78, 80, 80, 69, 59, 57, 67s
```

That is a **~4.7x slowdown** of a live experiment caused by a duplicate of a finished one. The duplicate is now gone and only the trainer remains.

## The finding nobody ordered: the retrain did NOT reproduce the first run

Same seed, same config, same data order - and the numbers are close on one instrument and **one point apart on the other**:

```
                        first run (pinned 2 cores)   re-run (unpinned 12 cores)
trainer line            23.02%                       23.03%          (+0.01)
independent evaluator   20.56%                       19.55%          (-1.01)
```

The trainer line reproduces to a hundredth. The independent figure moves a full point. The only difference is thread count: multithreaded BLAS
reduction order changes the last bits of every matmul, so an unpinned run is a **different model**, not the same model computed faster.

**Consequence for this project's rules:** a "same seed, same config" reproduction must also pin the same core count, or it is not a reproduction.
Thread count joins the list of things that must be held fixed in a paired comparison - alongside draws, windows and denominators.

## What happens to the published conclusion

It stands, and does not depend on the lost bytes: on the trainer instrument the arm reads **23.02 vs 22.88 (+0.14)** in the first run and
**23.03** in the retrain, and the independent instrument reads **20.56 vs 20.62 (-0.06)** then **19.55 vs 20.62 (-1.07)**. Every version is far below the
+0.5 bar, and the independent sign is negative in both. The falsifier still fires. What is lost is exact byte-level reproducibility of the first
run's independent figure, and this page now says so instead of citing a file that no longer holds those bytes.
