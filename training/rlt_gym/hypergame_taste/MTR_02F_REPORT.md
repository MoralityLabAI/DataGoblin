# MTR-02F: indexed lazy refinement under hard logical-work budgets

Registered protocol: 7ce4204736c2180f4dade8da446035612a4cc311.
Frozen implementation, weights and receipt: d7287da9ded2672940602563b0f43a0fcf23e8f1.
Prior MTR-02D/E files remain unchanged.

## Scope and method

32 training, eight calibration, 24 ordinary and 24 combined cubic-family/cost-shift OOD roots. All 88 content signatures are distinct. Each public deterministic root has 128 latent Boolean witnesses, a nonlinear bijective branch/payload encoding, 58 evidence-gated probes, positive integer priors and 7–9 research credits. Full, partial, branch and coarse witness predicates and abstention define terminal utility. The construction mechanism is carried forward from MTR-02E with fresh namespaces. These finite toy certificates do not certify source-paper theorems or exercise native strategic hypergames.

The policy index is constructed cold from public prefix gates. There are 31 prefix masks, 31 corresponding action-list headers and 58 action IDs: 120 logical entries (not bytes). Construction charges 236 units per arm (two per prefix and three per action). Traversal, candidate emission, legality filters and partition tests are charged. The full-scan Bellman oracle separately checks indexed legal-action IDs, costs, ordering and outcome masks at visited positive-budget states, then computes exact integer objectives using its own terminal aggregation. Every frozen incumbent is replayed explicitly over the 128 hidden witnesses with the appropriate prior weights.

Lazy leaves retain the terminal certificate and a clairvoyant upper bound. A single action job adds all its observation children; a constructive job computes one of two deterministic full feasible rollouts (route or information heuristic). The optimistic frontier follows the unresolved highest-upper-bound action and includes its descendant frontier or corresponding unexpanded action, plus unfinished constructive jobs at open visited nodes. A node's lower value is the best feasible base plan or expanded action with feasible child plans. Its upper value is the maximum terminal reward, expanded action upper value or unexpanded action clairvoyant upper value. Backups use exact unnormalized mass arithmetic and propagate in increasing remaining-credit order. Expansion/constructive updates are committed transactionally only after the charged graph and backup work completes. Prior committed incumbent remains feasible if the budget interrupts a job; any completed intermediate cache values are discarded when the arm terminates. Eager leaves instead construct the full base (terminal, all one-step-and-stop choices and both constructive rollouts) before committing; root initialization can retain improving feasible partial base plans. Analytic planning builds the same base and attempts up to two contingent refinements and offline replanning. All arms have independent cold policy caches. Action-only excludes constructive jobs. Work-aware gap uses a fixed public estimated job cost, not a learned cost predictor. Random is averaged across three seeds.

Logical work charges index construction, lookup/iteration/feature coordinates, witness and claim aggregation, partition tests, bound operations, graph updates, plan components, frontier and priority work. The immutable generator, evaluator oracle/replay and runtime overhead are excluded; these counters are an implementation-specific work proxy. Graph replay avoids constructing a separate policy tree, and is evaluator-only. Index/graph/cache occupancy is logical entries, not process memory. The Node CLI exposes a whole-campaign timing, CPU and RSS profile option; it has not been run here. It includes training/oracle/replay and cannot establish matched per-policy time without further instrumentation.

The two 20-feature deterministic ridge heads learn actual root lower gain and root certified-gap reduction per charged job work, scaled by 4096. Four sampled job presentations per training root (128 rows, 84 action and 44 constructive); 352 training jobs. Discovery has 28 positive targets (MSE 36.181164 vs constant 53.254656); certification 40 positive (109.566158 vs 174.375978). These are training fit diagnostics only. Neither head is a recurrent neural model. Training labels charge the refinement job itself, not the prior frontier selection or feature extraction; those operations are charged in evaluation.

## Calibration

The protocol selects the smallest tested budget with median action jobs >=8 and at least six of eight calibration roots completing an action. At the first candidate, 32768 units, median is 13 and all eight qualify. Primary 32768 and secondary 16384 were therefore selected before constructing any evaluation roots; no evaluation-root tuning. The previous experiment's primary budget completed zero expansions. Comparisons across MTR-02E/F are descriptive because the roots and search implementation differ.

## Evaluation

| Policy | Ordinary utility at 32768 | OOD utility at 32768 | Ordinary gap | OOD gap |
|---|---:|---:|---:|---:|
| Exact oracle | 30.143635 | 29.083869 | 0 | 0 |
| Eager | 27.837386 | 25.985707 | 7.162614 | 9.014293 |
| Capped analytic | 27.837386 | 25.985707 | 7.162614 | 9.014293 |
| Certification head | 27.545720 | 25.402373 | 7.370947 | 9.347627 |
| Lazy mass gap | 26.025972 | 25.109747 | 7.634906 | 8.416637 |
| Random (3 seeds) | 19.184879 | 20.274055 | 15.690121 | 14.392612 |
| Discovery head | 14.534865 | 15.383431 | 19.861206 | 18.823239 |
| Work-aware gap | 10.347842 | 10.351290 | 22.174684 | 21.730173 |
| Action only | 10.426386 | 10.247195 | 22.079690 | 21.742663 |

Every arm exhausts the work cap. No evaluation root closes its certificate gap or attains exact oracle utility at either budget. All 48 lazy mass-gap roots complete at least one action; mean action jobs ordinary/OOD 13.333333/12.541667. Work-aware gap completes 95.208333/95.416667 and action-only 101.541667/104.125, both with zero constructive jobs. Lazy mass gap completes 1.125/1.166667 constructive jobs, on 21/24 roots in each cohort. Certification completes 0.958333/0.958333 constructive jobs (23/24 and 22/24 roots) and only 0.25/0.833333 action jobs. The discovery arm completes 4.125/4.541667 action and 0.666667/1.041667 constructive jobs. Eager and analytic complete zero action expansions; the analytic incumbent remains a base plan on every root, so the capped comparator has not reached useful contingent refinement. Its upper bound is the initial bound. Eager/analytic equality is a result under the cap, not equivalence of the algorithms.

Certification versus mass gap utility +1.519748 [-1.036000,4.075495] ordinary, +0.292627 [-1.642202,2.227456] OOD. Certified-gap reduction versus mass gap +0.263959 [-2.321208,2.849126] ordinary, -0.930990 [-2.878784,1.016804] OOD. Certification versus analytic utility -0.291667 [-0.863333,0.280000] ordinary and -0.583333 [-1.374022,0.207355] OOD. Discovery versus analytic utility -13.302521 [-17.990419,-8.614623] ordinary and -10.602276 [-15.592990,-5.611563] OOD. Intervals are descriptive root-paired normal intervals on public development roots. False-publication probability at primary budget is zero for mass gap, certification, eager and analytic on both cohorts. Discovery has 0.003058/0.003899 ordinary/OOD. No learned head passes promotion.

At secondary 16384, mass gap/eager/analytic each earn 7 on both splits. Action-only earns 8.404605/7.879501; certification earns 0.583333/1.458333; discovery earns 1.458333/1.750000. The more expensive constructive jobs can exhaust the cap before committing any result. In particular, lazy scheduling is not a uniform utility improvement. The action-only control shows that more refinements alone do not constitute mathematical discovery. The work-aware rule preferentially buys cheap local actions rather than useful complete constructive plans.

## Audit

2,365,440 witness predicate checks, 88 nonlinear-selector checks, 48 exact oracle replays, 960 evaluation incumbent replays and root bounds, plus eight calibration incumbent replays (968 total). 25,207 monotone backups, 105,150 positive-budget legal-list comparisons, 713,896 indexed child comparisons. Separate receipt recalculation passed 4,893 arithmetic/consistency checks over means, intervals, per-arm metered work, calibration, bound inequalities and both saved weight arrays. Git blob IDs: receipt d0a82bafa7d37e94ff2177b83d1a8859fcf73346, weights cff7e261ef16051535fa729ba8eb33e02ef841d9, source 038fece24747c96522bd09dcd73be5e573d68074.

## Disposition and next candidate

Retain the indexed legal-action implementation, transactional single-action jobs and calibrated work meter. Do not promote either learned scheduler. The principal remaining resource cost is witness/terminal aggregation and constructive rollout work. The learned certification head largely selects the same expensive constructive job as the eager baseline. The next experiment should decompose constructive rollout into interruptible feasible subplans, using per-branch continuation certificates. Compare a deterministic constructive-first policy with a budget-conditioned portfolio over route, information, direct and contingent strategies. Charge each strategy's cold and incremental cost and record partial feasible incumbent value as a function of work. Fit a cost model and value improvement model on train-only censored job outcomes. Freeze fresh held-out construction families and calibrated budgets before evaluating. For proof certification, a tighter upper bound using the public feasible witness/cost structure is needed; a predictor of immediate root-gap reduction is insufficient if most work is spent on constructive plans. Native time/RSS measurement and independent private confirmation are still outstanding. No BPD activation mapping, neural recurrence advantage, native TRM/LDT integration or 1B training claim.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02f.mjs 20261127 1 > mtr02f.json
    node training/rlt_gym/hypergame_taste/mtr02f.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02f.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/mtr02f.mjs 20261127 1 --profile > mtr02f.json 2> profile.json

The experiment function ran in V8 and its exact source was committed. Node CLI and new branch CI have not been verified by this execution.
