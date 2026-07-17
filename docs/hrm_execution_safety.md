# HRM checkpoint execution safety

The default checkpoint task ID is `hrm-text-cycle-real-v1-ram4096`. It processes one verdict example at a time, flushes
every record, and checkpoints progress at least every 60 seconds. Operator-authorized caps are 4096 MB RAM, 50% CPU,
50 MB/s I/O, and 1800 seconds. Aborts are valid results.

`datagoblin hrm-preflight` performs only metadata and small-config checks. It does not import Torch, map weight contents,
or load a checkpoint. When the static footprint cannot fit, it writes `trainer_plan.json`, `events.jsonl`, and
`summary.json`, then exits nonzero.

`scripts/run_hrm_capped.ps1` is a fail-closed host gate. Windows Job Objects can provide hard memory and CPU caps, but the
documented Job Object I/O-rate function is unsupported on modern Windows clients. Therefore the gate refuses to start a
weight-loading child unless a future implementation supplies a verified cgroup/container I/O cap. Monitor-and-kill is
not represented as a hard cap. The official API limitation is documented by Microsoft:
`https://learn.microsoft.com/windows/win32/api/jobapi2/nf-jobapi2-setioratecontrolinformationjobobject`.

WSL2 was considered, but a health check must establish a responsive cgroup-v2 environment before use. DataGoblin never
edits the shared `.wslconfig`, shuts down an unrelated distribution, raises caps, or loads weights merely to test the
wrapper. Post-run cleanup is PID-scoped and must record RAM, commit, cache/standby, GPU applications, and lingering owned
processes.

After any future weight-loading child exits, `scripts/post_run_hrm_cleanup.ps1` performs that audit. It stops processes
only from an explicit owned-PID list and never purges global standby cache.
