# MTR-02D: Certified anytime allocation of mathematical research compute

Protocol: [MTR_02D_PROTOCOL.md](MTR_02D_PROTOCOL.md), registered at 03e4af75f6c728b592289aacf36041c4408452fd before the development run.
Implementation and frozen receipt: 34673fd262021cc5c15ffb4f5fc9e50dfd30322e.
Base: merged PR #97, b2dff47648ee607306d16903524bd1644f6e8cbd. Prior receipts remain unchanged.

## Mathematical model

There are 64 possible Boolean witnesses per root, positive integer prior masses, three nonlinear branch selectors and three payload coordinates. The 17 public research actions form a prerequisite DAG. Wrong representation probes can yield redundant partial observations. The certificate pool includes full witnesses (+30/-45), branch claims (+12/-18), branch-plus-one-payload claims (+18/-27), coarse claims (+7/-11), and abstention. All research actions cost credits. Budget varies from five to seven; each action can be used once. Redundant observations are legal because they can unlock descendants.

32 training roots, 16 ordinary and 16 cubic-family OOD roots. The OOD generator also changes costs and distractor outcomes, so this is a combined shift. All are deterministic public development roots. The four-presentation cap yields 128 training rows. A deterministic 18-feature ridge fit predicts normalized exact remaining value improvement (V*-L)/W/30 on training states. The model is a small priority head, not a recurrent neural RLT, and its target is a surrogate for marginal refinement gain rather than a measurement of one-expansion gain. Its training MSE is 0.004501 versus 0.068498 for a constant training-mean prediction. No evaluation oracle value enters its input or priority computation.

## Bound invariant

State s=(S,U,b) contains compatible witnesses, consumed action IDs and remaining credits. W(S) is integer prior mass. T(S) is the best terminal certificate reward in mass units. Each state starts with a feasible lower bound L from a library of 63 prerequisite-respecting path sets (including prefixes and global probes). The library is restricted and differs from the exhaustive open-loop MTR-02C comparator. The upper bound H is the maximum of T(S) and (30-min legal action cost)*W(S), or T(S) if no action is legal.

Expansion introduces every legal action and its possible observation children. The lower backup retains the best previously feasible plan or a legal action with feasible child policies. The upper backup maximizes T and action upper values. For each action a the backup is -cost(a)*W(S) + sum_o child value in mass units. Feasible policies give L<=V*, while the clairvoyant upper bound and Bellman induction give V*<=H. Each edge strictly increases the consumed-action count, so propagation is acyclic. The lower bound never decreases and the upper bound never increases. The optimistic policy defines the unresolved frontier; schedulers choose a frontier node using normalized gap, mass-weighted gap, depth-first, seeded random or predicted value improvement times mass.

A snapshot freezes the incumbent lower-bound plan and replays it on every hidden world. The exact root objective is used only to audit the snapshot. Shared public partition/terminal caches and the evaluator oracle are outside the scheduler's operation counts. Logical library evaluations, library scans, partition-cell visits, child transitions, backups, priority calls, frontier visits and DAG reuse are recorded. Expansion counts alone are insufficient to infer equal total compute. No actual time/RSS was measured.

## Policy results

Mean expected utility at fixed expansion budget (0/8/32/128); separate ordinary and OOD cohorts.

| Scheduler | Ord. 8 | Ord. 32 | Ord. 128 | OOD 8 | OOD 32 | OOD 128 |
|---|---:|---:|---:|---:|---:|---:|
| Optimistic normalized gap | 19.801595 | 23.943583 | 24.137396 | 16.551486 | 17.927323 | 19.341376 |
| Mass gap | 19.963966 | 24.175118 | 24.199766 | 16.687850 | 18.836027 | 19.483217 |
| Depth-first | 16.495761 | 16.840818 | 23.438538 | 13.493669 | 13.493669 | 17.135696 |
| Random | 18.534831 | 23.667060 | 24.199766 | 16.280564 | 19.090703 | 19.485837 |
| Learned priority | 20.318711 | 24.199766 | 24.199766 | 17.286663 | 19.483217 | 19.485837 |

Library-only starting value: 6.345134 ordinary, 6.132809 OOD. Exact oracle: 24.199766 and 19.485837. At eight expansions, learned minus mass gap is +0.354745 [-0.044995,0.754485] ordinary and +0.598813 [0.113567,1.084058] OOD. At 32 expansions it is +0.024648 [-0.023662,0.072958] and +0.647191 [0.115791,1.178590]. Learned versus random at 32 OOD: +0.392515 [-0.016568,0.801597]. These are descriptive root-paired normal intervals, unadjusted for multiple comparisons. Only one fixed random seed per root is used. At 32 expansions learned matches the oracle on 16/16 ordinary and 15/16 OOD roots; at 128 the utility matches on all 32 roots. Oracle equality is not a proof of optimality using the search's own bounds: at 32 none of its gaps close; at 128 only 12 ordinary and 3 OOD are certified closed. Mean remaining certified gap at 128 is 0.497125 ordinary and 1.273719 OOD for learned, versus 0.355365 and 0.897122 for mass gap. The learned priority is better at finding good incumbents here than at certifying them.

## Strong analytic baseline and work caveat

| Analytic planner | Ord. root lower | Ord. replan utility | OOD root lower | OOD replan utility |
|---|---:|---:|---:|---:|
| Restricted library B0 | 6.345134 | 6.345134 | 6.132809 | 6.132809 |
| One-step B1 | 10.417911 | 24.199766 | 9.480628 | 19.235837 |
| Two-step B2 | 18.162905 | 24.199766 | 15.944252 | 19.485837 |

B1/B2 replan after each actual observation and perform additional search work online. Their root lower bounds are distinct from their replayed utility. The reported baseline work counters are cumulative across B0, B1, B2 and their replays, in that order; they are not isolated per-run costs. With that caveat, the B1 cumulative including-replan mean library evaluations are 2173.5 ordinary / 1450.5 OOD; B2 cumulative 3852 / 2635.625. Learned 32-expansion mean library evaluations: 2302.875 / 1634.6875. Mass gap at 32: 2311.4375 / 1586.3125. At 32 OOD the learned scheduler spends 3.05% more library evaluations and 2.03% more partition-cell visits than mass gap (9744.5 vs 9550.25), and has additional learned feature/scoring cost. The root-fixed search plans and replanning policies have different computation schedules. The table therefore establishes no end-to-end speed or compute efficiency win over analytic replanning.

Learned 32-expansion false-publication probability is zero on both public cohorts. At 128, average ordinary expansions are 91.25 learned versus 105.5625 mass gap; early closure is allowed, but learned retains a larger average certificate gap, and the OOD difference in average expansions is negligible. No claim of reduced proof certification effort from these means.

## Verification and disposition

368640 finite witness checks, 64 nonlinear-selector checks, 32 exact oracle replays, 640 incumbent replays and 640 root bound checks, 7536 public partition mass checks. All passed. Separate receipt arithmetic validation passed 1634 checks over aggregate means, confidence intervals, work counts and snapshot bound invariants. The 18 stored weights are finite. The committed blobs are receipt 983d276c75c2387fd634d537be0eb6debf9e61b4, weights 47f9bdc20ab9a9d238a9db942b0b377b5f431fbf, source ec221f25104fa9c6677f685684a965078acc6660. The engine ran in V8; Node CLI invocation and full repository CI for this new branch remain unverified. No independent implementation of the new oracle, private holdout, actual BPD/MMS activation integration or measured wall-clock/RSS.

Disposition: retain learned priority as a development candidate. It leads the fixed-expansion scheduler means at small budgets, with the OOD gain versus mass gap positive in the descriptive interval. It does not establish a confirmed advantage over the strongest analytic planner, a recurrent advantage, or superiority at matched actual compute. Do not promote.

## Next test (proposed MTR-02E)

Freeze a stronger analytic replanner and isolate its costs by fresh cache and per-observation accounting. Compare policies under actual combined resource budgets (partition-cell visits, verification work, then measured CPU/RSS), including root preplanning, online replanning, learned feature extraction, repeated search and cache overhead. Use marginal root lower-bound gain and root certificate-gap reduction as separate targets; prioritize improvements in feasible policy and in proof closure separately. Include multiple seeded random controls and held-out deeper obligation constructions with measured analytic headroom. Register primary comparison/budget and adequate root count before execution. The reachable DAG can expose typed evidence events to BPD/MMS, with byte-accounted memory and exact-replay controls. No claim that this finite learned scheduler already provides native BPD or recurrent mechanisms.

## Reproduce

    node training/rlt_gym/hypergame_taste/mtr02d.mjs 20261127 1 > mtr02d.json
    node training/rlt_gym/hypergame_taste/mtr02d.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02d.mjs 20261127 1 --rows > rows.json
