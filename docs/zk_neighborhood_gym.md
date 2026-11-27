# ZKL-16: bounded neighborhood policies

Pinned SatsHedgeZero checker and proposal engine: 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb. Six arms use the
same exact polynomial verifier and lazy field columns: indexed frontier,
handcrafted row selection, trained static row ranker, trained feedback row
ranker, shuffled feedback scores, random row. Neighborhood = retained row and
its multiplier columns. Local ranking scans only the chosen row's pending
columns. The indexed reference maintains its exact global heap.

Training roots 16001–16008 on four families over F7/F11. Evaluation roots
16101–16108 and 16201–16208 on four other named families over F11/BN254, including
multi-hop product and inverse transport. Some algebraic motifs are related;
public roots are development evidence, not secret evaluation. Nonlinear
nuisance cycles have different offsets for train and test, and attach to
two core wires. All core assignments extend by setting nuisance wires zero.
Essential controls have explicit satisfying-retained/violating-target witnesses.
Each core row is rescaled, all wire IDs and row order are randomized.

Each teacher uses an independently verified full bounded-span certificate,
then replays its support. Reference support is nonunique; sampled negatives
exclude still-needed teacher rows. The feedback ranker uses explicit recurrent
per-row visits and decayed residual-size improvement. It is a small logistic
controller, not a BPD, TRM or LDT implementation.

Protocols: 24 or 96 offered columns; separate 96-column protocol with a common
10 ms process CPU deadline, checked at action boundaries. Setup, search and
independent verification are included in the deadline. A late verified
certificate is recorded but receives zero deadline credit. Atomic work may
overshoot; this is a cooperative deadline, not OS preemption. Record row and
column scores separately, symbolic/index terms, materialized coefficients,
elimination work, setup/search/verifier CPU and full timing. Each task shuffles
arm order deterministically. The hosted campaign uses a 512 MiB virtual
address-space limit and reports process peak RSS via GNU time; no per-arm
peak RSS is inferred. Training and full-reference work are reported separately.

The fixed protocol is 64 evaluation tasks, six arms and three protocols
(1152 runs excluding training and reference). No per-case Lean replay,
live-circuit optimization or production authorization.

## Hosted result

[Workflow 46009068947](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46009068947)
passed both evaluation blocks and targeted checks. Across 64 tasks:

| Arm | 24 columns | 96 columns | 10 ms credited |
|---|---:|---:|---:|
| Indexed | 37 | 38 | 38 |
| Row heuristic | 16 | 25 | 25 |
| Row static | 32 | 32 | 32 |
| Row feedback | 37 | 37 | 37 |
| Row shuffled | 30 | 43 | 42 |
| Row random | 29 | 46 | 46 |

All 16 essential controls remained uncertified. Indexed vs random at 96
columns used 0.287368 vs 0.343726 process CPU seconds across both blocks.
Peak RSS was 21308/21564 KiB for the two whole campaign processes. Random
row selection recovered 14/16 multi-hop product-chain cases vs 6/16
indexed. This motivates explicit exploration on a shared exact basis.
