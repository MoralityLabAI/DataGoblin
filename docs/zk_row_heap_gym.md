# ZKL-20: exact indexed exploration for the trained recurrent gate

ZKL-19's unweighted binary recurrent controller certified 92/96 at 96
columns versus 74/96 for its identical-weights reset ablation. The
10 ms protocol credited 84/96 recurrent vs 86/96 random neighborhood.
A candidate exploration action currently scans an entire sampled row,
even when exploitation is ultimately chosen. This experiment reuses
the existing global heap score entries in per-row versioned heaps.
The sampled row and RNG consumption are unchanged. Entries share
exact scores and tie breaks; stale entries are discarded on query.
Compaction filters existing entries, with a four-times-initial-row-size
trigger after updates. Record extra pushes, pops, compaction scans and
peak row heap entries. The extra index may increase RAM.

Pinned SatsHedgeZero checker and search basis: 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb.
Recreate the ZKL-18/19 unweighted 177-parameter RNN from the same
deterministic counterfactual training dataset (roots 18001–18008).
Both recurrent and matched reset models run with either old row
scanning or new row heaps. Same weights, same RNG and threshold.
Four prior controls remain. Compare entire action traces, certificates,
elimination counts, prepared coefficients, gate decisions and exploration
counts on uncapped-CPU protocols. Deadline-limited traces must share their
entire common prefix; time can change the stopping point. Every certificate
is independently verified. No change to proof acceptance.

Two fresh root blocks 20101–20108 and 20201–20208, all eight ZKL-19
families, F11/BN254, 128 tasks (96 certifiable and 32 essential).
Eight arms and three protocols (3072 evaluation runs), including
24/96 offered-column limits and the existing 10 ms cooperative
process-CPU deadline. All timings include setup and exact verification.
Training and reference costs separate. 512 MiB virtual-address cap and
whole-process peak RSS. No per-arm RSS claim, new architecture training,
generated per-case Lean replay or production authorization.
