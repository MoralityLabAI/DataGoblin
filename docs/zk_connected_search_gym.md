# ZKL-14B: connected nonlinear decoys and feedback search

Pinned proposal engine and exact identity verifier: SatsHedgeZero 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb.
Two evaluation blocks (14101–14108 and 14201–14208), four families each,
F11 and BN254 Fr. Train roots 14001–14008, four separate named families over
F7/F11. Some training and evaluation core algebra is related, notably inverse
transport; the decoy-return topology and exact families differ. These are
public development tasks, not blinded evaluation.

The six nuisance wires have constraints d_i(d_next + a_i c_i)=d_return.
Zero nuisance wires extend every core assignment. The training and test
cycle/chord offsets differ; each nuisance wire occurs nonlinearly and is
shared. The old graph peeling control must remove zero rows. Each essential
fixture has an independently checked retained-satisfying, target-violating
assignment. Wire permutation, row permutation and coefficients vary by root.

Eight arms: degree order, uniform random, original peeling, static overlap,
residual/monomial frontier, trained logistic static ranker, trained logistic
feedback ranker, shuffled static scores. Matched offered-column budgets of
24 and 96. The feedback ranker shares weights across visits and uses explicit
residual/frontier/visited-row state; it is not a trained TRM, LDT or BPD model.
Reference certificates supply nonunique positive labels; other columns are
sampled negatives, not logically necessary/irrelevant labels. Full reference
search work and training CPU time are reported separately.

All arms construct fresh lazy problems; scoring uses symbolic monomial
supports rather than field-column coefficients. Exact independent verifier
checks every proposed certificate. CPU process time includes fixture setup,
policy scoring, elimination and verification; no strict matched CPU-time
budget, memory measurement or general speedup claim. Also report allocation,
coefficient work and score counts. Full bounded span abstention does not
prove essentiality; the synthetic essential cases have explicit witnesses.

Run in a reviewed checkout with the pinned checker directory on PYTHONPATH.
For larger local runs use the existing CPU-only RAM watchdog. No production
optimization or per-certificate Lean replay is claimed.

## Hosted results

Pinned campaign commit: 93c1a4a8cd1e51c8a2f511fe0f76ff78336ad568.
[Hosted workflow 46006683995](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46006683995)
passed targeted Ruff, pytest, both 32-task evaluation blocks, and uploaded
the paired JSON receipts. The actual pinned checker files were used.
This is the authoritative result; local reproduction was not completed.

Across 64 tasks, 48 have degree-at-most-two certificates and 16 have
explicit essential-row witnesses. Each of the eight arms tries both budgets
(1024 paired policy runs, excluding reference and training calls).

| Policy | Certified at 24 | Certified at 96 | Offered columns at 24 | Process CPU seconds at 24 |
|---|---:|---:|---:|---:|
| Degree | 16 | 32 | 1254 | 0.0550 |
| Random | 0 | 0 | 1536 | 0.0508 |
| Peel | 16 | 32 | 1254 | 0.0535 |
| Overlap | 32 | 32 | 832 | 0.1997 |
| Frontier | 48 | 48 | 496 | 0.4258 |
| Learned static | 32 | 32 | 832 | 0.2287 |
| Learned feedback | 48 | 48 | 496 | 0.7476 |
| Shuffled | 0 | 1 | 1536 | 0.2274 |

No arm certified an essential row. Both dynamic policies solved every
square-transport case on both blocks. Each used 112 offered columns for
the 48 successful tasks (496 including essential controls). The frontier
scored 219904 candidates vs 192624 for learned feedback; the latter had
higher CPU cost. Both prepared symbolic supports for 74800 terms and
lazily materialized only offered field columns. CPU time is the sum of
process timings from the single hosted replay, not a matched-time trial.

Training used 32 tasks, 64 positive support labels, 3658 reference
columns and 4096 SGD updates per ranker, repeated deterministically
for each evaluation block. No learned advantage over the handcrafted
frontier was established. Next: indexed frontier selection with
charged CPU/verification budgets and independent harder topologies,
before training a larger recurrent architecture. No generated per-case
Lean replay or deployed circuit change.
