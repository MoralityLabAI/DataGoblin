# MTR-02G: interruptible constructive portfolios and information-relaxed proof bounds

Registered protocol: 0fec726ccf207f29bb52d13b980aefc8fbede5fa.
Frozen implementation, models, training rows and receipt: b60de34c0720d46fb8b0f552ef795b0eb5e9672a.
Prior MTR-02D/E/F experiments are unchanged.

## Scope and construction

32 training, 8 resource calibration, 24 ordinary and 24 OOD roots, 88 distinct generation signatures. Each root has 128 latent Boolean witnesses, positive integer priors, 7–9 research credits and 60 deterministic probes. The training families use quadratic-tree and coupled-payload encodings. OOD uses a cubic selector, shifted costs and a different cross-cut gate/probe mechanism. Prefix gates coexist with globally eligible and cross-cut action lists; the index charges construction, traversal, candidate emission and legality/partition tests. It stores 122 logical entries, not a byte-accounted process memory allocation. The exact oracle uses an independent dense terminal aggregation and full action scan; indexed legality and all corresponding outcome masks are checked at the oracle's visited positive-budget states.

A state is (compatible witness mask, remaining credits); an informative deterministic probe strictly shrinks the mask, so used informative probes need no separate history register. A reached graph node starts with the optimal immediate terminal publication or abstention and an optimistic upper value. A stream chooses a deterministic preferred action by route/arity per cost, information per credit, or immediate terminal expected reward. Its frontier traverses already expanded preferred actions. The immediate strategy stops where a one-step probe fails to improve the terminal. Each scheduled action expands all of its observation children with terminal fallbacks. The engine shares reached states across streams and may combine expanded arcs from different strategies into a new feasible plan. The lower backup maximizes immediate terminal and the sum of feasible child values less paid cost. The upper backup maximizes terminal and expanded-action upper values or unexpanded-action clairvoyant bounds, and intersects with the optional admissible relaxation. Propagation and graph updates commit transactionally, in increasing remaining-credit order; interrupted work consumes the budget and leaves the previous incumbent intact. Replaying a frozen root incumbent on each witness, using integer prior weights, equals its lower bound exactly.

The route and information stream arms also have optimistic-action fallback after their preferred stream has no job. Therefore the route comparison is between incremental shared-graph construction (with fallback) and a monolithic recursive route policy, not a pure execution-order-only ablation. Monolithic route/info return a plan only after the whole recursive rollout completes; before that they retain the immediate terminal fallback. They do not have MTR-02F's partial one-step library progress. The portfolio's constructive-first control enumerates available proposals for every stream before selecting the preferred one. This is deliberate shared candidate generation for portfolio comparisons, but incurs avoidable acquisition cost for that deterministic control. The route-only arm computes only route and optimistic proposals. Mass-gap uses optimistic action jobs alone; other portfolio arms also receive the relaxation proposal. These control differences limit causal claims.

The logical work meter charges public index construction, state/claim aggregation, indexed scans and observation partitions, proposals, frontier traversal, scoring, graph transactions and backups, and interrupted operations. Public root construction, evaluator oracle and replay, plus runtime allocation/dispatch overhead are excluded. Each arm starts cold. The Node wrapper supports a whole-campaign profile; it was not executed in this run. No actual timing, RSS or native BPD measurement is claimed.

## Training on interruptible jobs

128 public training presentations, four per root, sampled along mixed job trajectories. The preassigned per-job work quanta rotate through 512, 2048, 8192, 32768. A censored job consumed its quantum, committed no graph mutation, and recorded zero gain. Completed intermediate cache computations can be retained during that training trajectory. 12 rows are censored (2 of 103 action jobs, 10 of 25 relaxation jobs). There are 24 positive lower-bound gains and 29 positive certificate-gap reductions, all from action jobs. The cost model is a 22-feature deterministic linear log-work predictor fitted with one-sided squared loss for censored lower-bound observations. Separate 22-feature ridge models predict committed lower gain and gap reduction given the public remaining quantum. The scheduler ranks nonnegative predicted gain divided by predicted work, penalizing jobs predicted to exceed remaining work. These are linear heads, not neural recurrent models.

| Training target | Model loss | Constant prediction loss |
|---|---:|---:|
| Root lower gain | 2.044510 MSE | 6.178979 MSE |
| Root certificate-gap reduction | 2.057745 MSE | 6.229865 MSE |
| Log work | 0.253170 censored loss | Not fitted |

These are training diagnostics. The 128 sampled observations do not identify counterfactual benefits of unchosen jobs. The model target is immediate committed root improvement, so the utility of a sequence of individually unproductive prerequisites may be underestimated. Evaluation feature extraction and proposal acquisition consume additional work not included in each sampled refinement-job label.

## Train-only resource calibration

At the first candidate 8192 units, interruptible route completes a median 13.5 action jobs on eight calibration roots; all eight complete at least one. The primary budget is 8192 and secondary 4096, frozen before constructing evaluation roots. Calibration optimizes the action stream's resource feasibility, not the portfolio's ability to acquire expensive proposals. This distinction becomes material in the evaluation. Results across MTR-02E/F/G use different roots, arm definitions and work schedules and are descriptive.

## Primary results (8192 logical units)

| Policy | Ordinary utility | OOD utility | Ordinary certified gap | OOD certified gap |
|---|---:|---:|---:|---:|
| Exact oracle | 29.347304 | 29.206353 | 0 | 0 |
| Interruptible route | 13.347856 | 13.774651 | 21.652144 | 21.225349 |
| Mass gap | 7.000000 | 7.145578 | 26.646143 | 26.306979 |
| Learned discovery portfolio | 7.000000 | 7.000000 | 27.916667 | 27.916667 |
| Learned certification portfolio | 7.000000 | 7.000000 | 27.916667 | 27.916667 |
| Random portfolio (3 seeds) | 4.958333 | 4.126068 | 29.888889 | 30.762821 |
| Budget-conditioned portfolio | 4.666667 | 4.083333 | 30.083333 | 30.791667 |
| Constructive-first portfolio | 4.012039 | 3.653592 | 30.987961 | 31.346408 |
| Round robin | 3.791667 | 2.625000 | 30.958333 | 32.250000 |
| Interruptible info | 3.791667 | 2.625000 | 31.125000 | 32.291667 |
| Monolithic route | 0.784167 | 0.000000 | 34.215833 | 35.000000 |
| Monolithic info | 0.000000 | 0.000000 | 35.000000 | 35.000000 |

Route minus monolithic route: +12.563689 [11.046600,14.080779] ordinary; +13.774651 [13.279364,14.269938] OOD, descriptive paired normal 95% intervals. Only one ordinary monolithic route completes; no OOD route or monolithic info finishes. At 4096, the interruptible route yields 2.377563/3.452217 ordinary/OOD, versus zero from both monolithic constructors and every multi-proposal portfolio. Optimistic mass-gap earns 7 on both at 4096, showing that inexpensive immediate publication can beat constructive progress at a tighter cap. At 8192 interruptible route completes mean 13.875/13.25 action jobs, zero false-publication probability on both cohorts. No arm matches oracle utility or closes its certified gap at either budget.

Learned discovery and certification both commit one action job per root at 8192. Learned versus route utility: -6.347856 [-6.652260,-6.043453] ordinary and -6.774651 [-7.269938,-6.279364] OOD. They tie mass-gap utility on ordinary and trail it OOD. The learned certification gap is worse than mass gap by 1.270523/1.609687. No learned promotion. The ordinary learned portfolio spends a mean 7121 units on witness aggregation, versus 2676 for the route stream, at the same 8192 cap; OOD 7133 versus 2636. Information proposal estimation requests complete terminal statistics just to access child mass. Eager all-stream acquisition also charges immediate-proposal aggregation before selecting a single job. At the secondary budget, this prevents the multi-proposal arms from committing any action. The result identifies an implementation-level overhead mechanism; it does not isolate the cost of neural inference or prove that portfolios in general are ineffective.

## Admissible information relaxation

Let S be compatible witnesses, W(S) their total integer prior mass, b remaining credits, and T(S) the best terminal expected reward in mass units. Define E(S,b) to contain every probe with cost <=b and a gate intersecting S. Any probe used by a feasible continuation belongs to this set: its gate must contain the nonempty current subset of S. Ignoring future prerequisites and granting all observations in E for free forms a joint partition P of S. Witnesses within one joint cell generate identical observations for every feasible adaptive probe path, hence every attainable terminal partition is coarser than P. The optimum terminal expected reward can only increase when the agent may choose a claim independently for each finer cell (max of a sum is no larger than sum of maxima). Any nonempty research policy also pays at least c_min W(S), where c_min is the minimum currently legal informative first-action cost. Thus

    H_free(S,b) = max(T(S), sum_{C in P} T(C) - c_min W(S))

is admissible; if no first probe is legal, H_free=T(S). Intersect it with existing bounds. The upper bound relies on the deterministic-observation and finite public prior assumptions. It is not an external theorem proof or a general bound for stochastic sequential research.

Audit: 512 sampled positive-budget, legal-action-bearing exact-oracle states from eight evaluation roots (four per split); every relaxed value contains the exact optimum and is no larger than the original clairvoyant bound. 56/512 are strictly tighter. During actual capped scheduling, the budget-conditioned control completes one relaxation job per root, averaging 2250 units and zero tightening; random completes some and also obtains zero tightening. None of the learned or route arms completes a relaxation job at the primary budget. This separates mathematical admissibility from practical value per work. Do not promote the free-observation relaxation as a default expensive job on this workload.

## Verification and limitations

2,365,440 witness predicate checks, 88 nonlinear checks, 103,099 full/indexed legal-list comparisons, 743,402 child mask comparisons, 48 exact oracle replays, 1,248 evaluation incumbent replays and root-bound checks, 8 calibration incumbent replays (1,256 total), 5,019 monotone backup checks, 512 relaxation admissibility checks. All passed. A separate receipt audit recomputed 18,092 arithmetic and consistency checks over training caps/censoring/losses, calibration, paired intervals, operation sums and all saved monotone progress traces. Stored blobs: receipt 0a7ac56c7ce826a3f863afae4c9d5f82455ef36e, models ad70d87068c5a3d4e989e1712cf7382f5cf0368d, training rows d8e067028638f675e0ab537c8324d5f194dd5594, source 1c093c8d7c98bff7315e04a764ca744897a05527. The function ran in V8. Node CLI/profile and full hosted CI have not been exercised in this turn. These are public deterministic development roots; no private holdout, actual CPU/RSS measurement, external proof certification, native GPTStoryworld rollout or BPD/TRM/LDT activation mapping.

## Disposition / MTR-02H candidate

Retain interruptible route and its frozen progress traces as the strongest development constructor at the calibrated primary budget. Keep the learned failures and relaxation audit. Next, separate mass-only cache from terminal aggregation: a mass query must not build all claim bins. Make proposal acquisition itself a schedulable paid action; route first without eagerly enumerating all streams, and expand a second stream only when predicted incremental value exceeds its acquisition cost. Compare mass-only versus full terminal-stat lookup with an independent exact differential check, then benchmark interruptible portfolio decisions under equal hard logical work and measured per-arm CPU/RSS. Calibrate resource budgets on both route and the strongest portfolio's acquisition cost, before generating fresh evaluation roots, and require a minimum number of completed decisions per arm. Add option-level multi-job return/credit assignment rather than only one-job immediate gain. A cheaper proof upper bound can exploit branch-cost structure; test each bound's tightening per charged work against existing bound search. Do not reinterpret this public pilot as a successful learned policy.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02g.mjs 20261127 1 > mtr02g.json
    node training/rlt_gym/hypergame_taste/mtr02g.mjs 20261127 1 --models > models.json
    node training/rlt_gym/hypergame_taste/mtr02g.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/mtr02g.mjs 20261127 1 --profile > mtr02g.json 2> profile.json
