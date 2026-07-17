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

`scripts/run_hrm_capped_wsl.sh` implements the WSL2 path. Invoke it through the WSL root system manager; it enables the
runtime CPU/I/O subtree controllers when needed, starts a transient unit as the unprivileged owner of the staged model,
and verifies `memory.max`, `memory.swap.max`, `cpu.max`, `io.max`, and the model's backing block device before Python can
load weights. The model must be staged in WSL's block-backed ext4 filesystem: a checkpoint under `/mnt/c` or `/mnt/d` is
DrvFS/9p and cannot be governed by the unit's block-I/O controller. The launcher caps one smoke example at 4096 MB RAM,
zero swap, 50% CPU, 50 MB/s block reads/writes, and 1800 seconds. It emits JSONL resource samples, a summary, stdout and
stderr receipts, and uses CUDA only through the explicit `--device cuda` CLI option.

The released checkpoint currently requires a Transformers build from upstream `main`. Use an isolated environment and
pin an exact upstream commit; do not replace the system install or silently follow a moving branch. The launcher uses the
native, version-matched `hrm_text` implementation and does not opt into checkpoint-provided remote code. DataGoblin never
edits the shared `.wslconfig`, shuts down an unrelated distribution, raises caps, or loads weights merely to test the
wrapper. Post-run cleanup is PID-scoped and must record RAM, commit, cache/standby, GPU applications, and lingering owned
processes.

After any future weight-loading child exits, `scripts/post_run_hrm_cleanup.ps1` performs that audit. It stops processes
only from an explicit owned-PID list and never purges global standby cache.
