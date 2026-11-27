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
