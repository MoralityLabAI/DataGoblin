# MTR-02A result: long-horizon mathematical research decisions

Status: public development pilot. The [protocol](MTR_02A_PROTOCOL.md) was committed before execution (76af345fa2d3988c8c94c428bdb1909ed12af52d); source, per-root receipt and learned weights were committed together at eab5bd606c15c16ec423f01a0b0b95bbc847baff. Original MTR-01 and PR #6 receipts remain intact.

## Design and checks

A root lists 16 candidate mathematical witnesses and two nuisance states. Each witness is checked against box-LP feasibility and optimality, an invertible affine GF(2) system, or a triangular Boolean system. The latter is OOD only. There are two representations with paid preparation, noiseless and correlated noisy sketches, a public binary announcement conditioned by K=0,4,8,12,16 stochastic channels, a full-witness reward +16/-24, a coarse-consequence reward +4/-10 and abstention. Credits and visit limits vary from 6 to 8. Setup consumes credit and a visit but does not count as a verifier query. Policies choose when to stop; the publicly computable terminal macro chooses the highest posterior-value certificate. The information and rank heuristics are simple specified greedy controls; the rank heuristic is not an optimal minimum-cost basis planner.

Training 80 roots (40 each LP/affine); ordinary development 40 (20 each); OOD 40 (quadratic only, with shifted costs and observation law). Generation namespaces differ, and 160 generated content signatures are unique. Public deterministic seeds make these development splits, not sealed tests. The report bit is a conditioning scenario, not sampled at its marginal frequency. Exploratory hidden paths are sampled uniformly rather than by posterior weight; every target is an exact conditional action value. At most four presentations per root, 318 total. The 9 models have 386 trainable parameters each, common initialization seeds 19/37/73, 14 epochs, 4,452 Adam updates each. RLT learns the recurrent matrix and uses BPTT. Damping and pooled models use the same learned matrices on fixed decayed and averaged raw-history contexts respectively. All see current posterior-derived action/state features. Exact oracle Q values are used only as training targets.

The source checked 81,920 full witness/world pairs against the original finite mathematical definition, enumerated 156,514 exact DP states and compared the exact initial objective to explicit conditional hidden-world replay on 80 evaluation roots. Every policy was checked for legal moves and objective at most the oracle. The explicit replay and DP share certificate definitions; this is not an independent external verifier. Receipt audit independently recomputed 168 aggregate values/interval endpoints from the committed root rows and checked all nine parameter arrays for finite values and matching size. No Node CLI process, independent gradient audit, full Python suite or hosted CI was executed in this turn. The function bodies ran in V8 before commit.

## Result

| Policy | Ordinary expected utility | OOD expected utility |
|---|---:|---:|
| Oracle | 8.511024 | 6.241499 |
| RLT | 6.715272 | 2.813889 |
| Damping | 6.152337 | 2.508121 |
| Pooled | 6.606287 | 2.882231 |
| Information greedy | 4.702098 | 1.634275 |
| Rank greedy | 1.225178 | -2.570544 |
| Two-ply | 0.625 | 0.425 |
| Myopic | 0.075 | 0.050 |

The oracle minus two-ply gap is +7.886024 ordinary and +5.816499 OOD. The prior finite pilot had only +0.003689 on its public development roots. Different reward schedules mean absolute utilities between pilots are not directly comparable; the within-pilot headroom establishes complementarity.

| Comparison | Ordinary paired mean [descriptive 95% interval] | OOD paired mean [descriptive 95% interval] |
|---|---|---|
| RLT minus damping | +0.562935 [0.119297, 1.006573] | +0.305768 [-0.203775, 0.815311] |
| RLT minus pooled | +0.108985 [-0.125084, 0.343054] | -0.068342 [-0.480768, 0.344085] |

RLT vs damping query count ratios 1.0172/1.0254; RLT vs pooled 1.0080/1.0226 (ordinary/OOD). Mean false-publication probabilities: RLT 0.017492/0.059899, damping 0.022363/0.064111, pooled 0.020128/0.057597. RLT's OOD paired false-publication difference versus pooled is +0.002302 [-0.008479, 0.013083]. These are unconditional false-publication probabilities across hidden worlds, not conditional rates among publications. They are descriptive intervals, not confirmatory noninferiority tests. Actual verifier CPU time and memory were not instrumented.

Seed-level ordinary means (19/37/73):
- RLT 7.2252 / 6.8018 / 6.1187
- damping 6.5969 / 6.3914 / 5.4686
- pooled 6.8111 / 6.5154 / 6.4923

Seed-level OOD means:
- RLT 3.5583 / 3.3164 / 1.5670
- damping 2.8623 / 2.4667 / 2.1954
- pooled 3.2527 / 2.5733 / 2.8207

**Disposition: retain gym; do not promote learned recurrence.** The ordinary point estimate beats damping and pooled with a small extra query count, but OOD loses to pooled and the paired utility interval versus pooled includes zero on both splits. A single seed can reverse the ranking. The current experiment provides no confirmed recurrence-specific advantage.

## MTR-02B design implications

For a finite deterministic-probe world, let S be the set of currently compatible latent worlds, U the set of consumed/preparation actions, b remaining credits, t remaining visits, and W the fixed public prior conditioned on the announcement. Then (S,U,b,t; W) is sufficient for continuation value: legal actions depend on U,b,t, terminal expected utility depends on W restricted to S, and a probe produces the posterior partition S intersect outcome fibre. Induction on t gives a history-independent exact Bellman solution. Two histories that reach the same sufficient state have identical optimal continuation values.

The learned arms receive compressed summaries, not this full sufficient-state encoding. An observed history advantage may arise from the RNN compensating for feature loss. It does not imply memory is required by the task itself. MTR-02B should compare a state-complete learned scorer (posterior and availability masks, with permutation-safe representations) and a stronger costed-basis planner against the current history models. Match parameters, candidate access, training labels and budget. A frozen projection with matched bytes, a shuffled-memory arm, and explicit cache hit/miss counters distinguish BPD/MMS benefit from capacity. Use algebraic state merging to compare same-state different-history pairs; report disagreement separately from value error. For a memory-limited arm, hide the sufficient state and give all policies the same raw observations, then vary retained state size. Register independent ordinary and family-level OOD roots with a separate seed before fitting; treat public pilot as development only.

For research-grade mathematics, extend to actual proof-obligation DAGs with independent finite-checkable certificates and branching prerequisites; hold out entire construction mechanisms, not just an encoding. The exact optimizer is a teacher on small cases, not a scaling baseline on large instances. Native GPTStoryworld replay, BPD/MMS integration, the larger frontier-model campaign and 1B training remain open.

## Reproduce

    node training/rlt_gym/hypergame_taste/mtr02a.mjs 20261127 1 > mtr02a.json
    node training/rlt_gym/hypergame_taste/mtr02a.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02a.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/audit_mtr02a.mjs

Compare generated receipts and weights to the committed JSON files. The audit uses the committed receipt and weights as inputs, and does not independently regenerate trajectories. The audit receipt includes their git blob hashes.
