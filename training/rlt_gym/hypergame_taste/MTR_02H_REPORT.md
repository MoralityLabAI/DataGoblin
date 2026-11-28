# MTR-02H: paid proposal acquisition and multi-job research learning

Registered: b7a0336fe4b57a52d2262277f1125134ba7fcfd0.
Implementation and frozen pilot receipt: ff7395d0e68c19f0d61efa8e3d994693237c913d.
MTR-02D/E/F/G are unchanged.

## Scope and controls

32 train, eight calibration, 24 ordinary, 24 combined cubic/cost/cross-cut OOD roots, 88 distinct public deterministic roots. The 128-witness finite game and gate construction mechanism come from G, with fresh namespaces. All 12 arms use integer reward/bound arithmetic, the indexed gate engine and independent cold policy caches. Root construction, oracle/replay and independent verification are evaluator-only. The policy work metric charges declared logical operations, not elapsed CPU time. The two eager arms share the same all-stream mass-gap selection and differ in information-mass cache implementation (full claim aggregation versus split mass/count). Route and paid-option arms use the split cache. The route-first controller switches after two stalled route options; the two-stall criterion never fires at the primary budget. The ordinary and OOD route-first results are therefore identical to the single-route control. It is not evidence that switching adds value.

At an option boundary, the portfolio selects a stream before acquiring its next frontier proposal. Up to four jobs are executed transactionally; if the chosen stream lacks a proposal, the optimistic action stream is acquired as a charged fallback. An interrupted option retains completed jobs and incumbent. Cost, gain and gap targets include all selected-stream acquisition costs and completed/aborted job work. Option feature acquisition and selection overhead are charged at evaluation; their pre-option costs are not included in training return labels. Censored options can have positive committed returns. Option labels are observed selected outcomes, not counterfactual estimates for every stream.

Two 24-feature ridge heads learn root utility gain and root gap reduction, and a 24-feature log-work head uses one-sided squared loss for censored cost observations. 128 rows, four per training root, 28 censored; 444 committed training jobs. Gain training MSE 3.337633 vs constant 8.542519; gap 3.700719 vs constant 9.629737; censored cost loss 0.118504. 47 positive gain and 70 positive gap rows. Training uses a large outer cap with individual quanta 2048/8192/32768/65536; evaluation uses the selected small root cap. The outer-cap-relative feature and histories can shift in distribution. Training returns for the bound stream may include optimistic fallback action gains. These confounds matter: both learned heads select the bound option for every primary-budget option, 72 choices in each cohort. They commit about five action jobs and five bound jobs per root. Neither learns a useful mixed portfolio.

Calibration tests both route and route-first: at the first candidate 8192 units, both have median 13.5 action jobs and all eight roots complete >=2 options. Primary 8192, secondary 4096 selected before evaluation-root generation.

## Matched-work pilot

| Primary budget policy | Ordinary utility | OOD utility | Ordinary certified gap | OOD certified gap |
|---|---:|---:|---:|---:|
| Exact oracle | 29.753271 | 29.423605 | 0 | 0 |
| Route / route-first | 13.935210 | 13.535635 | 21.064790 | 21.464365 |
| Bound first then route | 11.069057 | 10.472804 | 23.157276 | 23.644425 |
| Information | 7.082848 | 7.138741 | 27.875485 | 27.777925 |
| Random options (three seeds) | 7.568978 | 7.155158 | 26.804608 | 27.192556 |
| Learned gain / certification | 7 | 7 | 26.684671 | 26.504804 |
| Eager split cache | 7 | 7 | 27.708333 | 27.625000 |
| Eager full cache | 2.041667 | 3.208333 | 32.875000 | 31.583333 |
| Round robin | 6.606792 | 6.922339 | 28.309875 | 27.869327 |

All primary arms use 8192 work units. No oracle tie or closed gap. Split cache versus full cache utility +4.958333 [3.658007,6.258660] ordinary and +3.791667 [2.366233,5.217100] OOD, descriptive paired 95% intervals. Learned versus route -6.935210 [-7.429121,-6.441299] ordinary and -6.535635 [-6.975442,-6.095827] OOD. Learned gap reduction versus route is negative on both cohorts. No learned promotion. Route and learned arms have zero mean false publication; bound-first has 0.004471/0.008866 ordinary/OOD. At secondary 4096, bound-first earns 7 on both cohorts, route 4.104873/3.079861, learned 4.375/5.25 and eager arms zero. The smaller-budget result suggests budget-dependent scheduling, but is secondary and public development only.

At 8192, eager acquisition consumes roughly 6901–7034 units per root. Route consumes 1094/1071. Paid acquisition eliminates repeated acquisition of unselected streams; however, a cheap selected stream alone does not ensure high solution value. Learned bound options consume about 2181/2064 units in completed bound jobs, tighten local upper values, and still leave worse root gaps than route.

## Mass/count differential and proof bound

512 sampled visited oracle states (eight roots), 512 identical information proposal IDs, 3665 direct mass/count comparisons, and exact terminal comparisons. Cold proposal work full 276676 versus split 190088, 31.29% reduction including index construction. This is a same-state operation-count audit, not a runtime measurement.

Let w_min and c_min be the minimum positive public world weight and probe cost. A nonsingleton child C with mass W has immediate terminal value at most max(24W,36W-90w_min): a full claim necessarily misclassifies at least one world, and all weaker correct claims pay at most 24. For a singleton use 36W. If remaining credits cover c_min, the upper bound on any nonempty continuation is (36-c_min)W; otherwise no probe is affordable. Maximize the immediate and nonempty alternatives. Apply this child relaxation to every legal first action, sum unnormalized child values minus first-action cost, and maximize with exact immediate terminal value. This is admissible under the finite deterministic model, and is no greater than the original clairvoyant first-action bound. Intersect it with current bounds. The prior G free-observation relaxation remains available as an independent comparator, not an input to learned labels.

| Cold bound audit (512 states) | One-step child-cost | G free-observation |
|---|---:|---:|
| Valid vs exact V* | 512 | 512 |
| Strictly tighter than original | 400 | 68 |
| Charged work | 207972 | 288959 |

Child-cost bound is strictly stronger on 345 states, free-observation on 44, equal on 123. Work decreases 28.03% relative to G relaxation. The learned arm completes about 4.7–4.8 strictly tightening bound jobs per root at primary. Local tightening does not imply root certification or useful search allocation. Treat the bound as an available paid refinement rather than an automatic default. The selected training data mixes fallback action and bound improvements; a future pure-bound control should separate them.

## Native cache primitive measurement

**Provenance correction (MTR-02I):** The v1 native timing receipt and numerical timing claims below lack verifiable execution provenance. They are preserved as a historical record but withdrawn as measured evidence. Use [the executed v2 replication](mtr02h_mass_native_replication.json) instead. The original native source was reproduced byte-for-byte (Git blob 8eb5354c988b880f6871481755085c2de09cac92) and run in fourteen fresh Node processes. V2 median elapsed 116.684754/105.106013 ms (full/split); median CPU 184081/170275 us; peak RSS 95516 KiB in both modes. Every checksum and cached-entry differential passes. Paired median elapsed/CPU reductions 11.43%/14.26%, with negative reductions in some pairs. No end-to-end policy or RSS improvement established. The original finite-game operation counts, policy receipts and arithmetic audits remain separate and unchanged.


Separate standalone Node v22.16.0/Linux x64 benchmark: 24 deterministic synthetic roots, 24576 mixed mass/terminal requests per process, seven alternating fresh-process pairs. Oracle generation and differential checks are outside the timed query loop; process peak RSS includes generation/checks and runtime. All runs have checksum 1938578 and all cached entries pass independent exact verification. This is a microbenchmark on a separate trace, not native execution of the full research-policy experiment.

| Metric | Full aggregation median | Split cache median |
|---|---:|---:|
| Query loop elapsed | 108.71344 ms | 108.101873 ms |
| Query loop CPU | 184004 us | 171689 us |
| Process peak RSS | 95508 KiB | 95508 KiB |
| Logical aggregation work | 1163135 | 754379 |
| Terminal cache entries summed across roots | 6039 | 3387 |

Paired median reduction elapsed 1.87% (range -4.98% to +15.68%); CPU 6.39% (range -7.46% to +21.52%). No clear wall-time or RSS advantage established on this small noisy trace. Split mode stores additional 6039 mass entries across roots. Retained entries summed across roots are not simultaneous process memory. Native results transcribed from local container runs; see full replicate receipt and source.

## Verification and disposition

2,365,440 witness checks, 88 nonlinear checks, 112035 full/indexed legal-list checks, 819026 child comparisons, 48 oracle replays, 1152 evaluation incumbent replays and bound checks, 16 calibration replays (1168 total), 11931 monotone backups. Separate receipt audit passed 30804 arithmetic/consistency checks over training caps, partial censoring, losses, primitive bounds, option/trace monotonicity, per-arm work, calibration and paired intervals. Stored original H blobs: receipt 39eb4795f0aa7ec3a003aa0eb94ae50ad21d6faa, models f2f3706b6159dc7cc512288fa5991de4a5936fb8, training rows 9255ef362b5e618bc556f9d8c413b831f7bbe4d1, source 494e7ff4dd6cdaca09b71058bc6340e123beb2f0. Pilot executed in V8. Native Node was used only for the separate cache microbenchmark. No full-campaign native CPU/RSS, fresh CI, private holdout, native TRM/LDT/BPD integration or external theorem certification.

Keep the split-mass primitive, paid acquisition and admissible child-cost bound. Keep all negative learned comparisons. Next MTR-02I: factor training credit between pure bound work and fallback actions; run paired train-only counterfactual option branches at identical state/caches/quantum, include route-first anchor and calibrated per-arm time, and learn delayed route progress relative to the deterministic route baseline. Freeze a budget-dependent portfolio and evaluate on a separate fresh family. Strong controls remain primary; avoid escalating model capacity until measured end-to-end search value improves.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02h.mjs 20261127 1 > mtr02h.json
    node training/rlt_gym/hypergame_taste/mtr02h.mjs 20261127 1 --models > models.json
    node training/rlt_gym/hypergame_taste/mtr02h.mjs 20261127 1 --rows > rows.json
    node training/rlt_gym/hypergame_taste/mtr02h_mass_native.mjs full 20261127 24
    node training/rlt_gym/hypergame_taste/mtr02h_mass_native.mjs split 20261127 24

Repeat the native commands seven times with alternating order, in fresh processes. Compare exact checksums and report paired timing ratios separately from operation counts.
