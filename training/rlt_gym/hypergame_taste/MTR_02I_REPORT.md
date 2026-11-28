# MTR-02I: paired option interventions and route-anchored scheduling

Registered protocol: a92b2de322bb7aa3179245aeef914b57b39d578a.
Frozen pilot: ba2844e18e4a0c7243421f7d96948a5f0108132a.
Post-hoc forced-route control: b3cba19e67a84989df16fd25eecea71568683523.
Preceding MTR-02D/E/F/G/H finite-game receipts remain unchanged.

## Design

32 train, eight calibration, 24 ordinary and 24 combined cubic/cost/cross-gate OOD roots, 88 distinct public signatures. The finite 128-witness generator and 60-probe mechanism are inherited from H with fresh namespaces. Positive integer prior masses, exact terminal claim rewards/penalties, prefix/cross-cut legality, constructive/action jobs and proof bounds. A pure bound option can only refine a bound; there is no action fallback inside this stream. Action options may use the optimistic fallback. Once a pure-bound stream has exhausted available jobs, it is blocked until an action expansion changes the graph.

For each training root, independently replay zero or one route option (prefix quantum 4096) in five cold engines. Compare exact serialized pre-intervention graph, cache entries, counters, prefix work, prior option history and bounds across all five engines before option feature acquisition. All 256 nonreference full-string comparisons passed. Compact prefix digests in the saved rows are diagnostic identifiers only; the actual equality check used full serialized strings. At each of 64 paired states, perform one selected option (up to four jobs, quantum <=4096), then continue route within a fixed 8192-unit horizon measured after the prefix. This horizon includes 70 units of feature acquisition; it does not charge evaluator-only graph serialization or replay. Complete committed jobs survive censoring. All 320 intervention policies were replayed over the 128 witnesses, matching their exact lower values. The five outcomes per state are grouped training observations, not independent samples; there are two paired decision sets and ten outcome rows per root.

Fit two deterministic ridge heads with 70 features (14 public features in each of five option-interaction blocks). Targets are signed differences in delayed feasible utility or root-gap reduction relative to the paired route continuation. A per-stream mean residual and within-stream permuted-label ridge are controls. Evaluation chooses route by default and switches only if a candidate residual exceeds +0.25. Scored policies acquire all five feature vectors and charge the full scoring loop before choosing a stream. Cheap deterministic controls skip that scoring loop. All options have quantum min(4096, remaining work); the same logical-work meter includes selection, acquisition, refinement and interrupted work. Root generation, oracle and evaluator replay remain outside policy costs. No native recurrent or activation mapping claim.

## Counterfactual training results

| Intervention | Immediate feasible utility gain | Delayed utility residual vs route | Delayed gap reduction residual vs route | Censored first options |
|---|---:|---:|---:|---:|
| Route | 3.777375 | 0 | 0 | 0/64 |
| Information | 3.688266 | -4.561024 | -4.561024 | 54/64 |
| Immediate payoff | 2.682665 | -5.166731 | -5.151106 | 64/64 |
| Optimistic | 6.182665 | -3.009316 | -2.040566 | 0/64 |
| Pure bound | 0 | -0.780148 | -0.258952 | 0/64 |

No positive delayed utility residual among 256 nonroute outcomes. Pure bound has positive delayed gap residual in 28/64 paired states; its immediate utility gain and action-job count are exactly zero. Its later route continuation is a distinct source of utility. Optimistic immediate gain exceeds route, yet its delayed value is lower at the fixed horizon. All continuation horizons consume 8192 units. These data identify the conditional effect of the specified interventions at the specified prefix, horizon, work schedule and generator. They do not identify general optimal strategy or future long-horizon effects.

Fit MSE: gain 1.099326 vs per-stream constant 1.169976, within-stream shuffled 1.218158 and global constant 5.293899; gap 1.115055 vs global constant 5.731732. These are in-sample diagnostics, not held-out prediction accuracy. The gain sample contains no profitable switch, so a test of positive switch discovery has weak support; the conservative route anchor is effectively a no-switch classifier on this distribution. Evaluation option scoring is more expensive than training's single selected-option feature cost. That is a deployment overhead charged in the pilot and isolated below, rather than an uncharged resource advantage.

## Calibration and evaluation

The first calibration candidate 8192 meets the preregistered rule: median route action jobs 12, all eight calibration roots complete at least two noncensored options. Primary 8192, secondary 4096 selected before constructing evaluation roots.

| Policy | Ordinary utility (8192) | OOD utility (8192) | Ordinary gap | OOD gap |
|---|---:|---:|---:|---:|
| Exact oracle | 29.778986 | 28.142038 | 0 | 0 |
| Route | 13.241905 | 13.192982 | 21.758095 | 21.807018 |
| Route-first | 13.241905 | 13.192982 | 21.758095 | 21.807018 |
| Pure bound first then route | 12.488910 | 12.440113 | 21.990079 | 22.035697 |
| Learned residual / gap / per-stream constant / permuted | 9.983440 | 10.088079 | 25.016560 | 24.911921 |
| Optimistic | 7.027405 | 7.000000 | 26.659008 | 26.499919 |
| Information | 6.438796 | 6.336172 | 28.561204 | 28.497161 |
| Uniform random (3 seeds) | 6.843053 | 7.049875 | 27.503338 | 27.111715 |

All arms exhaust their caps; no oracle tie or certified closure at either budget. Learned vs route ordinary utility -3.258465 [-3.672515,-2.844415], OOD -3.104903 [-3.576712,-2.633094]. Same deficits in root-gap reduction because upper bounds are identical for these route-only policies. Descriptive root-paired 95% normal intervals; no private or confirmatory inference. The two heads, constant and shuffled controls choose route on every option at both budgets. Route-first also chooses route only. The learned ordinary/OOD false-publication probabilities are 0.009083/0.006450 at primary; cheap route is zero on both. No learned promotion.

At secondary 4096, optimistic utility is 7 on both cohorts, route 2.270724/2.676315, and scored route 0.669190/0.992383. This repeats the small-budget value of cheap partial certificates. It does not establish a frozen optimal budget-conditioned policy.

## Post-hoc forced-route scoring overhead control

Regenerate the 48 existing evaluation roots at both work caps using the pinned I implementation. Force every choice to route after running the same feature and score loop with zero weights. Compare against the frozen learned and direct-route receipts. All 96 root-budget cases reproduce the learned integer objective, lower/upper, false publication, action/bound jobs, full operation counts and entire option trace exactly. Direct route also reproduces its frozen values. This directly isolates the logical selection overhead in this implementation; it is not a new held-out evaluation.

| Work cap / split | Direct route utility | Forced scored route utility | Extra mean selection work |
|---|---:|---:|---:|
| 4096 ordinary | 2.270724 | 0.669190 | 763.625 |
| 4096 OOD | 2.676315 | 0.992383 | 776.416667 |
| 8192 ordinary | 13.241905 | 9.983440 | 2097.916667 |
| 8192 OOD | 13.192982 | 10.088079 | 2113.958333 |

At the primary budget, approximately 25.6–25.8% of the cap is consumed by additional option selection work. Zero-weight scoring and the learned scheduler choose identical options under the same metered execution, so the observed utility loss is attributable to this charged feature/scoring overhead. This is an operation-count ablation, not a measured CPU comparison. The 70-dimensional interaction representation is sparse but the code charges dense extraction/scoring. A cheap compiled gate or sparse evaluation may lower cost; neither has been tested here.

## Native-cache provenance correction and actual replication

The earlier MTR-02H v1 timing figures lack independently verifiable execution provenance and are withdrawn as measured evidence. They remain in the historical receipt with a warning in H's report. This turn reproduced the committed standalone cache benchmark source byte-for-byte (Git blob 8eb5354c988b880f6871481755085c2de09cac92; SHA256 ee031849fa40ef5823aec2b068a3219cb95e3542eaf8adc478094fb6a4e28961) and actually ran seven alternating fresh-process pairs in the local container under Node v22.16.0/Linux x64. Each process handles 24576 deterministic mixed mass/terminal requests over 24 roots. All checksums 1938578 and independent cached-value checks match. The elapsed and CPU timers cover the query loop; RSS is whole-process peak including evaluator generation. Full raw numeric replicate results are in mtr02h_mass_native_replication.json.

| Median | Full | Split |
|---|---:|---:|
| Elapsed ms | 116.684754 | 105.106013 |
| CPU us | 184081 | 170275 |
| Peak RSS KiB | 95516 | 95516 |
| Aggregation work | 1163135 | 754379 |

Paired median elapsed reduction 11.43%, range -4.99% to 14.99%; CPU 14.26%, range -15.21% to 27.13%. Split cache is faster on most measured pairs, but this is a small primitive trace with runtime noise. RSS is identical in the reported environment. These figures supersede the earlier unverified native timing figures; no native end-to-end policy performance claim.

## Verification and next candidate

2,365,440 witness checks, 88 nonlinear checks, 104158 full/indexed legal-list comparisons, 757530 child comparisons, 48 oracle replays, 1480 policy replays (320 training, eight calibration, 1152 evaluation), 1152 root bounds, 26556 monotone backup checks, 256 identical-prefix comparisons, 282 pure-bound credit checks. Separate receipt arithmetic/consistency audit passed 35357 checks over paired residuals, root caps, finite weights, work sums, intervals, options and traces. Saved pilot blobs: receipt 813915ec08da42b0aff92109438b5db822498ff8, models c876d7595560d413cd85a7e5882a36721b5c2dea, training rows cf2719e32493cbe43886d2f7eb513ec8887edc4e, source 2e2e2539607ac9b36e9fb5e368c35f6436fa6a91. The pilot and overhead audit function bodies ran in V8. Full policy Node CLI, new hosted CI, native TRM/LDT/BPD integration and private holdout remain outstanding.

Keep paired intervention training and pure-bound credit isolation. Do not promote the learned selector. MTR-02J should freeze a cheap budget-conditioned baseline (optimistic for small cap, route for larger cap) and compile a sparse abstaining deviation gate whose default has near-zero inference overhead. Before training, design fresh construction families with route-inadequate states and genuine positive counterfactual switch support, while preserving honest zero-support controls. Calibrate positive support and decision budget on training roots only, include stratified route failures and paired longer-horizon option returns, and use a state-complete deterministic gate and no-op scoring ablation. Compare total work and measured native per-arm time, rather than claiming benefits from fit loss or number of expansions. A private confirmatory run should freeze the gate and primary objective before evaluation. No evidence of native recurrence-specific gain.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02i.mjs 20261127 1 > mtr02i.json
    node training/rlt_gym/hypergame_taste/mtr02i.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/mtr02i_overhead_audit.mjs > overhead.json
    node training/rlt_gym/hypergame_taste/mtr02h_mass_native.mjs full 20261127 24
    node training/rlt_gym/hypergame_taste/mtr02h_mass_native.mjs split 20261127 24

The overhead audit is a post-hoc source-pinned transformation of I, rerunning only the frozen public evaluation roots. Repeat the native primitive commands seven times in alternating order in fresh processes.
