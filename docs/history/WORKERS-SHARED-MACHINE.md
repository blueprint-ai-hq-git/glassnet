# Two workers share this machine - measured 2026-11-07 19:08

This is a fact, not a plan. Both loops are live, both use the same GPU and the same `lens/runs/` directory, and
nothing in either loop knows about the other beyond `GPU-CLAIM`.

## What is actually running

    glassnet-wake.service        "start GlassNet wake"   (agent woken by glassnet-wake.timer)
    glassnet-chain-8m-resume     the 8M ungated arm (this loop) - train_chat.py pid 54314, resumed 19:01:38
    glassnet-res-long_compare-20261107-185304   stable_vs_plain.py pid 9872, started 18:53:04
    glassnet-liveview.service    live viewer
    glassnet-boot.service        exited

Evidence for the other loop, not inference: `stable_vs_plain.py` (pid 9872) has PPID 2206 = `systemd --user`, i.e. it
was started by a unit, and its output `lens/runs/LONG-COMPARE.txt` is being appended every ~2 min in a different
experiment family (models "PLAIN + stab", HELD_OUT ~0.04-0.05 at step 4,000 - not one of the chat ladder's arms).
It started 11 seconds before this loop's first post-boot command, so it is another agent's work, woken by its own
timer.

The same other loop also produced the unquoted `/bin/bash -c` launch that failed with rc 126 at 18:11, and an 8M
attempt pointed at `prepared_dialogue_only_honest` on cores 2,3 - i.e. it is working the same research area with
different conventions. That is the collision risk worth naming.

## Rules that keep them apart (apply to both)

1. `lens/runs/GPU-CLAIM` is the only mutex. Take it before training, release it on exit, honour a live claim.
2. A claim is stale ONLY if its `pid:` is not in `/proc`. Then it may be taken over, and the takeover must be logged
   with the dead pid (implemented in `run_datascale_8m_nogate_resume.sh`). A live owner is never pre-empted.
3. One trainer per configuration axis at a time. Two trainers in different experiment families (chat ladder vs
   stable-vs-plain) coexist fine at 377 MiB total - the 5 GB cap is not at risk - but they share CPU: load average
   5.24 at 19:08, and each chat epoch is pinned to two cores.
4. Nobody deletes or rewrites another loop's logs, checkpoints or docs. Archiving needs sha256 proof, in either loop.
5. A number is published only from a log the publishing loop can read end to end, with a line number. Numbers from
   the other loop's runs are quoted only with its log path and are never merged into the chat ladder's tables.

## What this means for the ladder

The chat ladder's arms are unaffected in validity by the other loop's work: separate configs, separate tags, separate
logs. The one caveat is throughput - epochs are timed while another process is running, so per-epoch seconds here are
upper bounds on a quiet machine, not the engine's best. Stated in the results rather than silently averaged.
