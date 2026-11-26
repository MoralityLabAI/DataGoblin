# MTR-02C report: nonlinear research decisions and analytic residuals

Protocol: [MTR_02C_PROTOCOL.md](MTR_02C_PROTOCOL.md). The protocol was committed before the pilot at b01ba4f7a27027a0c48b4921b49052b85fc92a18.
Pilot: bd7e6282e7401bf04bb5eec238daf8255c43e405. Post-hoc contingent control: f35381d45ef8b43cdb6508cafc474b45add9f9c9.

## Scope and outcome

Public development, 48 training roots, 24 ordinary and 24 cubic-family OOD roots, 192 training presentations capped at four per root. Nonlinear Boolean evidence, two prerequisite-gated representations, competing full/branch/coarse claims, setup and probe costs. Three seeds for each of five 4,272-parameter arms; 1,728 optimizer steps per fit. The learned models receive the public compatible-world mask, used-action mask, prior, probe tables, remaining resource state and analytic action features. The terminal macro and legal-action filter are shared. No nested strategic hypergame chain was exercised in this controlled pilot.

129,024 finite certificate checks, 96 nonlinear-selector checks, 37,463 action-wise analytic lower-bound checks, 48 exact policy replays and 384,105 memoized oracle states. The analytic features were computed in an instrumented evaluator that also queried exact Q to check lower bounds. Exact Q is not a model input; it is used as the training target and as an audit quantity. These counts do not represent measured deployment compute.

| Policy | Ordinary mean | OOD mean |
|---|---:|---:|
| Oracle | 17.592261 | 15.093467 |
| Analytic open-loop replan | 16.110202 | 14.833215 |
| Direct learned | 16.899159 | 14.186050 |
| Stateless residual | 16.769823 | 14.269038 |
| Matched damping | 16.766028 | 14.356361 |
| Pooled history | 16.686112 | 14.299206 |
| RLT residual | 16.308505 | 14.243385 |

RLT minus analytic on ordinary: +0.198303, descriptive paired 95% interval [-0.664627,1.061233]; OOD: -0.589830 [-0.944776,-0.234884]. RLT ordinary minus state: -0.461318 [-0.804738,-0.117899]. RLT also loses to both history controls in both split means. The registered recurrence promotion gate fails. RLT false-publication probabilities are 0 and 0.001796; query means 4.825422 and 4.588389. There are 192 same-state history pairs per split, including repeated constructed pairs. RLT action disagreement is 0.003472 ordinary and 0.024306 OOD; stateless controls are invariant. This is a diagnostic of the learned readout, not evidence that the problem requires recurrence.

## Stronger analytic control (post-hoc)

Let a state contain the compatible-world set S, used-action set U and remaining resources. W(S) is its unnormalized prior mass. Let T(S) be the best expected terminal reward in mass units. For an affordable prerequisite-closed set of future probes E, let P_E(S) be its observation partition. The nonadaptive value B0(s) maximizes T(S) and

    sum over C in P_E(S) of T(C) - W(S) * cost(E).

Define the contingent lower bounds recursively:

    B_(k+1)(s) = max(T(S), max legal a [-W(S)*cost(a) + sum_o B_k(s_(a,o))]).

The open-loop plan can be split at its first legal action and continued with its remaining set in every outcome; redundant suffix probes can be omitted, and prerequisites are setup actions. Thus B1 >= B0 for this finite model. Monotonicity of the Bellman operator gives B_(k+1) >= B_k, while feasibility gives B_k <= V*. Replanning at each decision attains at least B_k: induct on the remaining budget using the selected action and the monotone lower bounds on each child. Realized expected utility of receding-horizon policies need not increase with k.

A separately coded evaluator regenerated all 48 evaluation roots. It matched the original exact oracle and open-loop policy utility on each root. It passed 11,297 contingent bound checks, 24,289 partition-mass checks and 192 full policy replays.

| Control | Ordinary utility | OOD utility | Ordinary exact ties | OOD exact ties |
|---|---:|---:|---:|---:|
| Open-loop replan | 16.110202 | 14.833215 | 6/24 | 10/24 |
| Contingent 1 | 17.580513 | 14.988055 | 13/24 | 13/24 |
| Contingent 2 | 17.577182 | 15.034832 | 13/24 | 17/24 |
| Exact | 17.592261 | 15.093467 | 24/24 | 24/24 |

The 1-step control closes 99.21% of the ordinary open-loop-to-oracle gap; 2-step closes 77.47% of the OOD gap. The 1-step lower bound is 17.559429 ordinary and 14.817970 OOD; the 2-step lower bound is 17.575493 and 14.842076. Contingent 2's ordinary replan utility is slightly lower than contingent 1's, consistent with the distinction between lower-bound monotonicity and policy utility. These are post-hoc results on the same public roots, not a preregistered or independently held-out comparison. Planner wall-clock cost remains unknown.

## Cache locality-shift experiment

Six evaluation roots, 480 requests per root per phase, two phases with disjoint hot sets. Each of 13 arms receives 5,760 queries (74,880 verified cache-arm answers). Exact fresh per-query expansions were computed for distinct states and charged on misses. 32-byte cache entries include value, keys, work and metadata. Frequency/utility arms reserve 25% of bytes for a decaying sketch, which reduces entry capacity. Scan and victim comparisons and metadata operations are counted; evaluator trace and oracle tables are outside the cache budget.

| Bytes | Admission | Stable expansions | Shift expansions | Total |
|---:|---|---:|---:|---:|
| 0 | None | 37123 | 35166 | 72289 |
| 1024 | LRU | 7770 | 9457 | 17227 |
| 1024 | Frequency | 13651 | 20327 | 33978 |
| 1024 | Work | 6230 | 6260 | 12490 |
| 1024 | Utility | 6783 | 7785 | 14568 |
| 4096 | LRU | 5576 | 3969 | 9545 |
| 4096 | Frequency | 5576 | 4669 | 10245 |
| 4096 | Work | 5576 | 3714 | 9290 |
| 4096 | Utility | 5576 | 4669 | 10245 |
| 16384 | All four | 5576 | 3699 | 9275 |

Work admission at 1 KiB saves 82.72% charged expansions relative to no cache and 27.50% relative to LRU. It has more metadata operations than LRU (802,951 vs 496,440), so no speed claim is justified. A 25% frequency-sketch overhead is not worthwhile on this trace; sketch hash collisions, replacement scoring and capacity are confounded. The same 4 KiB expansion counts for frequency and utility are a receipt observation, not general policy equivalence. Explicit CPU timing, RSS and index/cache construction costs are required before extrapolating to MMS RAM allocations.

## Audit and disposition

A separate receipt recalculation passed 631 arithmetic checks over means, paired normal intervals, history diagnostics, cache totals and byte counts. The committed 1,344,238-byte weight blob was retrieved using its Git blob ID and parsed: 15 models, 4,272 finite weights each. Blob IDs: source 1128a68b343fb386853582f2358c7be9b47a2bce, receipt 22b32849c3e5e2bb7b5a63151eaaf616867ee3d6, weights 5158a6a762910095ce476e6819d06855ce655941, contingent receipt 2e395d5131e859eece6b42c006adadd57c9e801a.

Functions ran in V8 before committing. Node CLI wrapper execution, Python integration, full CI, gradient checking, native BPD/TRM/LDT integration, CPU/RSS measurement, and private holdout evaluation remain open. The toy certificates do not certify the source papers.

**Disposition: retain the gym, the stronger contingent control and work-only cache as development baselines; no recurrence promotion.** The next protocol should use contingent planning rather than open-loop search as its analytic comparator. It should increase adaptive depth and introduce expensive branching proof obligations where affordable limited-contingency planning demonstrably leaves headroom. A residual may prioritize branches for verification, with a safe fallback to the best verified plan. Any improvement claim requires fresh family-level roots, matched actual search/verification cost, and measured time/RAM. Avoid tuning the current public roots into a success.

## Reproduce

    node training/rlt_gym/hypergame_taste/mtr02c.mjs 20261127 1 > pilot.json
    node training/rlt_gym/hypergame_taste/mtr02c.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02c_contingent.mjs > contingent.json

The contingent script reads the committed pilot receipt and reconstructs the roots from their IDs. The pilot exports its capped presentations with --rows.
