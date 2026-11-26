# Math Taste V2 (2026-11-27): finite mathematical discovery and delayed evidence

Status: public development pilot, no trained neural RLT, no frontier theorem certification. Successor to [PR #6](https://github.com/MoralityLabAI/DataGoblin/pull/6). Base: master 9aa8fc009f0449894b53a70e99e7d22d79621525, which contains the earlier campaign files. The original report and receipt remain unchanged.

## Sources and evidence policy

- [Cogentic collection, pinned 4a7bc8a](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers): read the index and nine paper READMEs; the six November 25 papers announce Lean formalizations for a later companion release. Treat asymptotic results as published claims pending independent replay, not as ground truth labels.
- [OpenAI November 24 release, pinned 3344da5](https://github.com/openai/math/blob/3344da5/history.md): nine manuscripts, with Lean formalizations of selected results. Distinguish an exact checked statement from the broader manuscript claim. No new Lean compilation is asserted here.
- Cogentic descriptions are attributed to their authors; the finite generators and checks below are original, independent toy constructions. The paper algorithms are not reproduced.

| Source family | Proposed research action | Independent acceptance |
|---|---|---|
| [Bimatrix Nash](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/bimatrix_nash_equilibrium) | Switch mixed strategies, request payoff evidence | Rational regret at most 1/5 |
| [Log-rank](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/logrank_conjecture) | Switch between row sums, column sums, entries, rank | All entries of a proposed rectangle agree; exact integer elimination |
| [Edge reconstruction](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/edge_reconstruction) | Compare invariants with costlier deck evidence | Canonical graph under all vertex permutations |
| [Fictitious play](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/fictitious_play_convergence) | Detect stalled convergence, reset or change algorithm | Exact finite empirical duality gaps (proposed) |
| [VC compression](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/vc_sample_compression) | Buy or compress labeled evidence | Finite consistency and reconstruction (proposed) |
| [Prophet inequalities](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/subadditive_prophet_inequalities) | Allocate budget across arriving projects | Exact finite dynamic program and offline comparator (proposed) |
| [Integer factoring](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/integer_factorization), [deterministic factoring](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/deterministic_factoring_l34), [finite-field DLP](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/discrete_logarithm) | Select or audit algorithms on tiny generated instances | Exact arithmetic witnesses, scope and complexity-model checks (proposed) |
| [LP and tensor geometry](https://github.com/openai/math/blob/3344da5/history.md) | Search primal-dual certificates; choose decompositions | Exact feasibility and strong duality for finite LP; tensor identities and operation counts (tensor proposed) |

The 4x4 Boolean-matrix and 5-vertex graph tasks do not instantiate the claimed asymptotic bounds. The LP tasks do not establish strongly polynomial complexity. The 2x2 game solver does not implement MaximinKiteNE.

## Pilot contract and results

Each root is a finite set of publicly listed possible mathematical worlds; truth is latent. A binary announcement is conditioned on K nested stochastic channels (K=0,2,4,8,12,16). This is a chain proxy, not native GPTStoryworld strategic replay. A public seed picks the report bit as a conditioning scenario rather than sampling reports by their natural frequency. Exact posterior masses are BigInts. Each research action reveals a deterministic observation at cost 1, 2 or 3. Publication of a valid certificate earns +10, invalid earns -15; abstention earns zero before research cost. The three-candidate native subset is an artificial shuffled restriction. All policies see the same root and face identical credit limits; the native policy has deliberately restricted candidate access. The evaluator enumerates all hidden worlds conditional on the report. The receipt counts policy/root/budget evaluations, not individual hidden-world rollouts. Finite certificate checks include repeated checks across roots.

The Bellman recursion for posterior mass W(S) and unnormalized value U(S,b) is:

    U(S,b) = max(0, max_c[25 W(S ∩ A_c) - 15 W(S)],
                 max_q[-cost(q) W(S) + sum_o U(S ∩ q^-1(o), b-cost(q))]).

A_c is the set of possible worlds where candidate c is mathematically valid. The policy commits when its best terminal action beats the available probes. The two-ply and myopic controllers replan after each observation. The exact oracle and native oracle were checked by explicit per-hidden-world policy replay. Nash checks also have a separate floating-point arithmetic cross-check; LP witnesses are cross-checked by enumerating the small integral vertices. Graph canonicalization checks relabellings, not an independent graph-isomorphism package.

384 roots, 6,912 policy-root-budget evaluations, 44,224 candidate certificate checks, 2,304 exact DP replay comparisons, 1,493 synthetic presentations (maximum four per root). All are public development data. No test split or generalization claim.

| Budget 3 | Expected utility | Expected credits | False publication probability per root |
|---|---:|---:|---:|
| exact | 8.725903 | 1.269955 | 0.000166 |
| two-ply | 8.722214 | 1.273644 | 0.000166 |
| myopic | 8.293963 | 1.697443 | 0.000344 |
| native exact | 4.194570 | 1.307789 | 0.000596 |
| no probe | 1.382784 | 0 | 0.065522 |
| fixed order | 7.918789 | 2.016041 | 0.001252 |

The two-ply oracle gap is 0.003689 (descriptive paired normal 95% interval [0.000463, 0.006916]); the myopic gap is 0.431940 [0.400520, 0.463360]. The native-pool restriction explains a large artificial gap. The shallow optimum ceiling motivates an explicit delayed-evidence stress case. These are conditional expected utilities, not realized Qwen rollouts. The intervals are public-development diagnostics, not preregistered significance claims.

## Delayed evidence stress: exact LP certificates from linear sketches

Let h be uniform in F_2^d, b_i=1+h_i. The hidden LP is max sum_i x_i subject to 0<=x_i<=b_i. Its unique optimum is x=b and dual multipliers all one certify it. Each research query returns one parity of h, at unit or double cost. A randomized invertible cheap basis is guaranteed. Costly parity queries are placed first in the fixed-order control. Before d independent equations the compatible solutions form an affine fibre of size at least two. Every proposed full optimum has posterior correctness at most 1/2, making its expected reward at most -2.5; abstention dominates. This proves the optimal initial value is 0 at budget <d, and 10-d for budget >=d with d<=8. Rank-per-credit greedy attains it. One- and two-step policies abstain at the root for d>=3. This is a designed complementarity negative control: recurrent learning must be compared to the algebraic rank baseline. It does not establish a learned recurrence advantage.

The stress receipt is generated by direct enumeration of certificate checks and parity signatures for d=3,4,6,8 at budgets d-1,d,d+1. Each dimension has 16 generated probe banks shared across budget conditions; there are 64 distinct roots. Every budget is a paired comparison.

## Next registered proposal (MTR-02, implementation pending)

1. Add Python adapter and independent rational verifier for the public API. Use canonical, content-derived root IDs; separate source-family IDs, root splits, probe-bank generators and secret holdout seeds. Reformatting does not create new roots. Existing public pilot rows are train/development only. Add adversarial certificate mutations, hypothesis shifts, instrumented verifier cost, and a strict no-oracle-input audit. Reproduce source statements before ingesting any proof dependency. Distinguish raw mathematical observations from supplied posterior abstractions.
2. Construct multi-stage proof-obligation and information-acquisition worlds with genuine branching, lossy observations, expensive alternate representations and multiple competing tasks. Freeze an independently implemented oracle and include full-horizon rank, greedy lookahead, adaptive submodular/information-gain controls where applicable. Choose horizon >=6 and require a preregistered positive gap over two-ply in development; include unsolvable, misleading and native-pool-miss cohorts. Native GPTStoryworld replay should use actual actor-specific goals and counterfactual legal moves.
3. Fit the RLT correction head using training-root-only verified transitions. Baselines: fixed-depth, one-step/greedy, two-ply, learned stateless scorer, matched damping, shuffled evidence, matched-size no-recurrence, analytic rank/DP when applicable. Same candidate generators, wall-clock/CPU accounting, verifier budgets and visit limits. Store failures as evidence and expose only public observations. Evaluate by paired root utility at fixed credits, realized utility, calibrated false-publication rate, verifier calls, wall-clock, and p95 RAM; report by K, horizon, family, and OOD condition. No promote if only native-pool widening explains the effect.
4. Interface with the existing research-controller routing and TRM/LDT modules through a versioned observation/action schema; BPD/MMS can cache reached-state evidence and alternative certificate candidates under explicit RAM limits. Test 0/64/256/1024 MiB cache allocations with exact-answer equivalence and charge cache construction and cold misses. Fit all representations on training roots only; the memory control stores equal bytes of shuffled/irrelevant evidence. Separate cache performance from recurrent-policy gains.
5. Frontier model rollouts: start with 128 independent fresh roots per provider/model where access exists, retain raw outputs, tool actions, verifier receipts, token and cost logs. Sample against exact and strong heuristic policies. Export accepted demonstrations, rejected trajectories, and synthetic repairs separately, with no more than four total presentations per root. Add a 1B-model experiment only after verifying a positive recurrent advantage on fresh roots; hold out entire mathematical construction families. No frontier model calls or 1B training were made in this refresh.

Proposed primary gate: positive paired utility improvement over matched damping and stateless two-ply-equivalent controls on predeclared held-out roots at equal credit budgets, with a one-sided upper confidence bound on additional false-publication probability of 0.005 and no more than 5% extra verifier compute. Freeze thresholds and sample-size/power justification before training; numbers here are proposed design choices, not achieved metrics. Use cluster bootstrap by generation root, separate seeds and family-level OOD tests. Keep prior run and this run's receipts immutable.

## Replay

From the repository root, using standard Node.js:

    node training/rlt_gym/hypergame_taste/math_taste_pilot.mjs 20261127 16 > pilot.json
    node training/rlt_gym/hypergame_taste/math_taste_pilot.mjs 20261127 16 --rows > train-dev.jsonl
    node training/rlt_gym/hypergame_taste/lp_rank_stress.mjs 20261127 16 > stress.json

The two embedded JavaScript functions were executed in a V8 environment before committing the exact function bodies and receipts. Node CLI wrappers, full Python suite, and hosted CI were not run here. The public PR contains the source and receipts, not a private holdout archive.
