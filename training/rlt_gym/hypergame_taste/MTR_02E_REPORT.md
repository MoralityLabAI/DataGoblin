# MTR-02E: hard-work-budget discovery and certification

Registered protocol: 33129f90b83485ea25e4e44bdf08110752613981.
Executed pilot: e3c5ee565ec561a1e87da925ce786f44eed7dd10.
Post-hoc indexed legality audit: e2142cb77d446cff416d0a61424c608fa551ba26.
Base: MTR-02D on PR #100. Prior reports and receipts remain unchanged.

## Scope

Public deterministic development roots: 48 train, 24 ordinary, 24 combined cubic-family/cost-shift OOD. Each root has 128 possible Boolean witnesses, four nonlinear selector coordinates, three payload coordinates, 58 deterministic evidence-gated probes, positive integer prior weights and 7–9 research credits. The public branch/payload encoding is bijective. The terminal library contains full, partial, branch and coarse certificate predicates and abstention. No nested actor hypergame, native GPTStoryworld rollout or frontier theorem verification was exercised here.

The state is (compatible witness mask, remaining research credits). A legal informative deterministic observation strictly reduces the compatible set, making previously used probes uninformative. The exact Bellman oracle uses independent terminal aggregation and exhaustive legal-action enumeration; its policy is replayed on each latent witness. Feasible lower-bound plans are replayed independently of the search graph. The optimistic upper bound allows clairvoyant full-witness payoff less the minimum legal probe cost, or immediate terminal reward. DAG backups preserve monotone lower and upper bounds. A transactional expansion stages child nodes, parent links and all ancestor bound changes before committing; an interrupted expansion cannot change the incumbent. Cached intermediate evaluations may remain allocated on interruption, and are not used to resume within the stopped arm.

The shared base library includes the terminal, all one-probe-and-stop plans and two deterministic constructive strategies. The analytic comparator constructs depth-one/two contingent plans and attempts complete offline receding-horizon policies under the same hard work cap. Completed improving root candidates survive an interruption. The optional uncapped online replanning comparator was not executed. Each arm starts cold. Root construction and public truth/probe tables are shared evaluator inputs; oracle computations and evaluator-side replay are outside the policy work budget. Policy cache lookups, world visits, claim checks, legal-action scans, partition tests, plan components, bound work, graph work, frontier visits, feature/scoring work and backups are charged. Units are implementation-specific logical counters, not CPU instructions or a wall-clock proxy with validated weights. The hard cap is enforced before charged operations and includes interrupted work. Unmetered language/runtime overhead, allocation and immutable preprocessing still require actual timing and RSS measurements.

Two deterministic ridge heads (16 features each) predict actual root lower-bound gain and root certificate-gap reduction respectively, divided by charged expansion work and scaled by 4096. Each training root contributes four sampled expansions. The 192 rows contain 53 positive discovery and 119 positive certification targets. Training MSE: discovery 0.00567206 versus constant 0.00653682; certification 0.02432720 versus constant 0.02733174. These are fit diagnostics, not held-out predictor accuracy. The models are linear priority heads, not neural recurrent RLT or BPD models. Random controls use three independent fixed seeds per root, averaged within root.

## Registered results

| Budget | Ordinary utility, all arms or range | OOD utility, all arms or range |
|---:|---:|---:|
| 8192 | 7.000000 | 7.000000 |
| 32768 (primary) | 17.381032 | 16.101203 |
| 131072 | 28.855988–29.154282 | 28.304780–28.602128 |

At 8192 and 32768 no scheduler completes an expansion. At the primary budget only one ordinary and two OOD roots even finish initializing the search graph. The initial feasible plan is retained; the priority comparison is uninformative. This is a failed primary design, not evidence of equivalence of the scheduling rules. No budget was retuned or promoted after examining the development results.

At 131072:

| Policy | Ordinary utility | OOD utility | Ordinary certified gap | OOD certified gap |
|---|---:|---:|---:|---:|
| Exact oracle | 29.716274 | 28.967306 | 0 | 0 |
| Analytic capped offline | 29.154282 | 28.602128 | 5.845718 | 6.397872 |
| Mass gap | 28.882623 | 28.304780 | 5.514229 | 5.259700 |
| Work-aware gap | 28.882623 | 28.304780 | 5.521081 | 5.235177 |
| Random (3 seeds) | 28.868443 | 28.309131 | 5.536630 | 5.431446 |
| Discovery head | 28.855988 | 28.318644 | 5.545771 | 5.781088 |
| Certification head | 28.859874 | 28.312224 | 5.542718 | 5.773242 |

Both heads lose to the analytic comparator on both cohorts at this budget. Discovery versus analytic utility: ordinary -0.298294 [-0.479830,-0.116758], OOD -0.283484 [-0.441254,-0.125714]. Certification versus analytic utility: ordinary -0.294408 [-0.484387,-0.104429], OOD -0.289904 [-0.444878,-0.134930]. Certification versus work-aware gap (positive means smaller certified gap): ordinary -0.021637 [-0.151629,0.108355], OOD -0.538065 [-1.276751,0.200621]. These descriptive paired normal intervals are unadjusted, on public development roots. All arms have zero mean false-publication probability at the largest budget; none of their own bounds closes on any evaluation root. The analytic comparator's upper bound is the initial upper bound; its reported gap is not a search-refined certification objective.

The analytic arm consumes a mean 131072 units ordinary and 129178.291667 OOD; two OOD runs complete before the cap. Every search scheduler exhausts the cap. Only 11/24 ordinary and 11/24 OOD roots complete any expansion in each learned/deterministic search arm at 131072. Mean expansion counts ordinary/OOD: mass gap 2.291667/14.125; discovery 2.208333/7; certification 2.041667/6.791667. Larger expansion count does not imply better policy utility. No learned head passes the registered promotion requirement.

## Audit and diagnosis

2,580,480 witness predicate checks, 96 nonlinear-selector checks, 48 exact oracle replays, 1152 feasible incumbent replays and root bound checks, 10409 monotone backup checks, 528 training expansions. A separate receipt recalculation passed 5372 checks across aggregate means, intervals, per-arm work sums, upper/lower invariants and both 16-element weight arrays. Stored blobs: pilot source 1eccf473792a8ae52e9cf9e7a3b9e95de87e7d62, receipt a2a168624fe2b69ba3ebf6585a18c12f3635e93d, weights 8f5b444ea91f68e57d0a21208a9e65f0953c4f24.

The ordinary mass-gap arm at 131072 spends approximately 47109 units on full legal-action scans, 23034 on partition checks, 29544 on witness aggregation, and 13052 on lookups. The OOD distribution is similar. This identifies a structural initialization and action enumeration cost, not a measured runtime bottleneck. The primary budget failure motivates an indexed exact lookup and incremental library initialization rather than increasing the budget after the fact.

## Post-hoc exact prefix-index audit

All non-global actions are gated by a prefix of the four-bit branch label. If S is nonempty, the only prefixes satisfied by every witness in S are the ancestors of its longest common prefix. Precompute the binary prefix masks and action lists. At a node with compatible mask m, intersect it with the left child mask. An empty intersection selects the right child, a full intersection selects the left child, and an intermediate intersection terminates traversal. Emit the actions on that unique path plus global actions in their original order. Then apply the same cost and informative-observation filters. This preserves exact legal actions, costs, outcomes and deterministic tie order. Maximum candidate count is 13 rather than 58. The index is built from public probe definitions.

The independent post-hoc evaluator regenerates the 48 frozen evaluation roots, compares complete legal lists including each child mask at every positive-budget state visited by its exact recurrence, and independently reproduces the committed oracle objective. It checks 133644 visited states, including 105336 positive-budget states, 303120 legal action instances and 718950 child masks. All 48 oracle values match exactly.

| Counter | Full scan | Prefix-indexed |
|---|---:|---:|
| Candidate scans | 6109488 | 1193338 |
| Prefix tests | 0 | 375612 |
| Index reads | 0 | 529686 |
| Candidate emissions | 0 | 1193338 |
| Observation partition tests | 2642908 | 2642908 |

Candidate scans reduce 80.4675%. Including prefix tests, index reads and candidate emissions, lookup work reduces 46.1170%. Including identical partition checks, combined work reduces 32.1913%. These are explicit operation counts for the visited exact-oracle states, not total scheduler work, measured time, memory or a forecast of the learned-arm effect. The indexed evaluator is a separate exact implementation; the original registered scheduler receipt was not modified or rerun with it.

## Disposition and next registered candidate

Keep the negative primary result, the hard work meter, the transactional proof-bound engine and the verified prefix index. Do not promote either learned head. MTR-02F should use the prefix-indexed lookup inside all policy arms, account for index construction and storage, and compare eager initialization against a lazy feasible-library evaluator (terminal and one-step first, constructive strategies as separately scheduled refinements). Freeze new generator namespaces and budgets after a train-only resource calibration, before evaluation. Charge the same measured operations, then repeat with real CPU time and RSS plus independent replay. Include the capped analytic comparator, three random seeds, deterministic proof-gap/utility schedulers and both learned targets. Store counterfactual gain/work data only from training roots. If analytic search remains nearly optimal, increase genuinely branching proof obligations rather than adding unnecessary recurrence.

Full native Python/TRM/LDT/BPD/MMS integration, private holdout, independent proof-level validation and wall-clock/RSS measurements remain outstanding. The functions ran in V8; Node wrappers and new branch CI have not been verified by this run.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02e.mjs 20261127 1 > mtr02e.json
    node training/rlt_gym/hypergame_taste/mtr02e.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02e.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/mtr02e_gate_index.mjs > gate_index.json

The gate audit reads the committed original receipt. The audit records logical operation counts only.
