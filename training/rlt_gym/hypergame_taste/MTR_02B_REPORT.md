# MTR-02B: state compression, learned policies and cache scheduling

Public development only. Registered protocol: 2b9138579adfe12a4e00387c2ef261c2dbeb8681. Pilot code, receipt and weights: 473f6b2b4767a9f5f596f3b2d64a99abc942fa1b. This report adds a post-hoc algebraic oracle audit, clearly separated from the registered comparison.

## Learned policies

40 train roots, 20 ordinary and 20 OOD, 160 capped training presentations. 36 models: widths 4/8/16, four arms, three seeds. Parameters 974/1962/4034 within each width; 1120 updates per fit. Public seeds, unique generation signatures, no sealed holdout. Full-state token explicitly encodes all compatible-world and used-action bits, while the raw-history arms receive only public root features, prior weights, last action/outcome and budget. All arms share a posterior-based terminal publication macro and externally filtered legal actions. The latent coordinate count is a model bottleneck; it is not process RAM. Evaluation reuses deterministic decisions through a simulator-side cache, not an additional model input. Full-state models are small bottlenecked learned baselines, not exact state-complete solvers. Same-width weights and training budgets match; memory widths are capacity tiers, not matched across tiers. The split design correlates ordinary family with budget (LP=7, affine=8); OOD changes encoding and gates/costs together. No isolated causal claim for those factors.

| Width | RLT ordinary | Damping ordinary | Pooled ordinary | Full ordinary | RLT OOD | Damping OOD | Pooled OOD | Full OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8.41146 | 8.604271 | 8.619753 | 7.933947 | -0.421780 | -0.516013 | -0.353934 | -2.159047 |
| 8 | 8.448310 | 8.631471 | 8.484452 | 5.860387 | -2.298064 | -1.891506 | -2.122607 | -3.854366 |
| 16 | 8.116556 | 9.148243 | 9.327419 | 4.127771 | -1.512885 | -1.858313 | -1.321617 | -3.823830 |

Exact oracle: 13.2 ordinary, 4.4 OOD. Rank greedy: 13.2 ordinary, 0.060452 OOD. Two-ply abstains. No RLT width beats both learned history controls in both cohorts. Width 16 RLT minus pooled ordinary: -1.210863 [-2.343925,-0.077801], descriptive paired normal interval. All RLT OOD means are negative. Disposition: no promotion.

Certificate checks 117760; DP expansions 112808; exact explicit weighted oracle replays 40. The 400 same-state pair instances include repeated setup-order pairs across chosen latent worlds. 14400 model/pair checks of Bellman identity. Exact and full-state readouts are order invariant; RLT action disagreements range 21.0–25.7% ordinary and 11.8–19.3% OOD. Learned Q disagreement is a symptom of lossy or order-sensitive representation, not a proof of task memory requirements. Full-state policies receive a small bottleneck and have no guaranteed permutation invariance between distinct roots. A separate aggregate recalculation over the committed receipt passed 494 arithmetic checks (receipt blob d9de9e91b7500bfa2ab9868c9f8df82c9a851c2b); the arithmetic audit does not independently rerun SGD. No native BPD map, gradient audit, Python CI or measured RSS in this pilot.

## Exact cache experiment

Eight roots (four from each evaluation split). Each trace has 180 distinct nonterminal exact states, repeated in three passes with immediate duplicate requests. 5400 queries per cache arm; 10 arms, 54000 checked answers. Across the same trace, per-state miss work was measured by a fresh local DP, with identical within-query memoization, then charged on each simulated cache miss. No wall-clock speed or total memory claim. 32-byte fixed-size records (8-byte world mask, 8-byte exact integer value stored in float64, 16-byte key/age/work/valid fields), no separate index; a scan counts occupied record comparisons. Only the cache array-buffer size is byte-accounted. Caches are cleared per root. Work-aware admission compares the incoming miss's measured work to the LRU victim's measured work. Unrelated-key entries use an impossible key bit and cannot produce hits. The global scan overhead is substantial and is counted separately.

| Capacity | Policy | Hits | DP expansions charged | Occupied key comparisons | Peak live bytes per root |
|---|---|---:|---:|---:|---:|
| 0 | none | 0 | 85918 | 0 | 0 |
| 4096 | LRU | 2102 | 51299 | 486602 | 4096 |
| 4096 | work-aware | 2658 | 29516 | 458592 | 4096 |
| 4096 | unrelated | 0 | 85918 | 625152 | 4096 |
| 16384 | LRU/work-aware | 3960 | 22713 | 487697 | 5760 |
| 65536 | LRU/work-aware | 3960 | 22713 | 487697 | 5760 |

4 KiB work-aware saves 65.65% charged expansions vs no cache, 42.46% vs equal-byte LRU. This is a narrow reuse workload, and no extrapolation to GiB-scale memory or end-to-end speed is justified. The precomputed trace-cost table is evaluator infrastructure, excluded from the byte-limited cache. A native implementation must charge construction, index storage, eviction, actual query compute and process RSS. Larger capacities saturate because the trace has only 180 distinct states per root.

## Post-hoc exact affine representation and stronger analytic baseline

Every probe in this pilot is a binary linear functional on the encoded witness y and nuisance bit z. Encoding is bijective in all three families. Let v=(y,z) in F2^(d+1). An observation adds an affine equation a·v=o. Canonical reduced row-echelon form of the augmented equations determines the fibre exactly; the public weights restricted to that fibre determine all posterior quantities. All weights are strictly positive. Two consistent systems describe the same fibre iff their reduced augmented bases agree. For ambient dimension D=d+1, a fixed packed canonical array uses at most D(D+1) logical bits: d=4 uses 30 vs a 32-bit mask; d=5 uses 42 vs a 64-bit mask. These are logical bits, not measured memory savings. A full witness is determined iff every coordinate functional of y lies in the observation row span (equivalently every vector in the homogeneous kernel has zero y component). Rank alone is insufficient when the nuisance bit is unresolved.

The separate affine implementation independently regenerates the public roots from the committed IDs and seed; it uses packed augmented RREF keys, enumerates consistent worlds to compute exact integer posterior masses and verifies transition mass conservation. It compares its exact DP objective to all 40 original bitset-oracle root values. It also searches the set of prerequisite-respecting affordable probe sequences for the cheapest full-identification certificate, then replays that sequence on every latent world. It does not call the original pilot's evaluator. This is an independent implementation of the finite model, not an independent proof of any external theorem. The stronger planner is post-hoc, and must be frozen before a future test.

- ordinary: 20 roots, affine oracle mean 13.2, safe costed-basis mean 13.2, exact ties 20/20, 50880 affine DP states, 13210 distinct fibres, 5091 planner states.
- ood: 20 roots, affine oracle mean 4.4, safe costed-basis mean 4.4, exact ties 20/20, 39560 affine DP states, 13470 distinct fibres, 4462 planner states.
- 40 root-level exact DP comparisons, 26680 fibre checks, 180742 transition checks, 1408 safe-plan witness replays.

## Next experiment (proposed)

MTR-02C: use the canonical sufficient state as a typed evidence object and an explicit algebraic rank/coverage planner as the default comparator. Learn when a more expensive nonlinear representation or a proof obligation has value; train a residual over a competent analytic controller. Construct nonlinear probes and non-affine constraints so this affine algorithm is no longer an exact solution. Compare ordered recurrent memory against canonical state, a set-equivariant architecture, and pooled features at matched model and actual resource budgets. Use a memory utility target for cache admission from measured continuation cost times reuse probability; compare against byte-equal LRU, frequency and work-only policies under independent traces with locality shift. Freeze before testing, include same-state history invariance and counterfactual evidence deletion. Extend cache accounting to CPU timing, indexing overhead and process RSS. No 1B training, private holdout or native GPTStoryworld integration was done.

## Replay

    node training/rlt_gym/hypergame_taste/mtr02b.mjs 20261127 1 > pilot.json
    node training/rlt_gym/hypergame_taste/mtr02b.mjs 20261127 1 --weights > weights.json
    node training/rlt_gym/hypergame_taste/mtr02b_affine_audit.mjs > affine.json

Both function bodies were executed in V8 before committing. Node wrappers and full repo tests were not run.
