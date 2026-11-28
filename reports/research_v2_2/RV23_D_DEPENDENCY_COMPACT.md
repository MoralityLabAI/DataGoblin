# RV23-D: dependency-mask compactification

Parent RV23-C 3fbf8367 passed both hosted workflows (827 passed, 16
skipped, clean lint). CPU-only synthetic public experiments. Frozen
native upstream 7fa19c90372b13d71676e6fef13819d2a18b9be3.

## Mathematical constructions

Let S be a closed selected set, R the unselected separators for one pair,
d its positive deficit, and A_i=C(i) minus S for i in R. Every feasible
completion for the pair contains the union of at least d distinct
separator closures. Equivalently minimize cost(U) over unions U of A_i
such that |U intersect R| >= d. One direction follows because any d
separator choices cover d separators. In the other direction choose
d separators in U: closure of each is contained in S union U, since
the latter is closed. Their residual union is a subset of U and
cannot cost more. This equivalence handles nested closures and cycles
(co-selection constraints, not a feasible chronological schedule).

Exact quotient: graph states are residual union masks, edges U -> U OR A_i,
absolute cost nondecreasing with strictly positive item costs on strict
growth. Best-first expansion by union cost, deduplicate exact masks,
and stop on the first union satisfying the deficit. This gives the
same exact lower bound as combination enumeration. The within-world
cache key is (remaining separator mask, deficit, sorted distinct
residual closures); costs are fixed within the solve. Equal signatures
imply identical quotient graphs and objectives. No cross-world reuse.
Cap at 256 entries; retain only exact completed bounds.

Union atlas: enumerate all original need=2e+1 separator combinations
for a pair, deduplicate full dependency union masks, and evaluate
min cost(U minus S) at later selected states. Any pair-feasible
completion contains at least one original need-set closure; any
such closure joined to S covers the pair. Thus the atlas gives the
same exact residual bound, even if some separators are already
selected. Identical original separator masks share the atlas; cap
retained atlas masks at 4096. A nonretained atlas can still be used
for the current bound. No unbounded memo.

Cheap control bound: sort the residual closures' total prices and
their prices restricted to nonseparators. Any completion must contain
d separators, at least the sum of d cheapest direct separator costs,
and at least the d-th smallest nonseparator closure price. These
cost classes are disjoint. It must also pay at least the d-th
smallest total closure price. Max these two lower bounds. Independently
verify this new witness by set-based reconstruction in v2 proof
checker. Never sum across pairs. Old v1 proofs remain replayable.

## Work and evaluation

Same RV23-C proof-producing search for all arms. Cover, forced, original
all-pair closure enumeration, cheap control bound, residual quotient
with/without exact cache, full-union atlas. New exact arms first
compute cheap control bound and avoid exact refinement if it closes
the node. 20,000 exact work terms per solve (quotient candidate OR
attempts, or atlas construction combinations plus scan terms).
Legacy enumeration counts combinations separately. Partial
minimizations never authorize pruning; fallback to previous bound.
Also reserve original combination count for every newly selected
exact witness, conservatively capped at 20,000, so the independent
enumerating proof checker can replay it. Checker uses original
combination enumeration for closure witnesses, not quotient/atlas.
CPU includes solver, proof production, cache accounting; verification
CPU separately measured. Shallow retained bound-cache estimate includes
memo/atlas keys and values, may double count shared integers, excludes
transient heap, proof, common solver and RSS. Count caps not RAM caps.
One randomized timing per arm/root/cap, paired bootstrap over roots,
no multiplicity adjustment.

Fresh 24 native frontier roots, 16 each shared/transitive and bundled
structured 16-item worlds, 72 roots; caps 64 and 65536, 7 arms,
1008 searches. Bundled cases deliberately use co-selection cycles.
Exclude all 176 previous A/B/C raw root identities. No structural
isomorphism claim. Full-cap exact oracle and certificate replay
required on all roots, plus native reference comparison on native
worlds. Independent tests compare residual and atlas bounds with
explicit combinations on small closed states; verify cheap bound
inequality, cached result, exact full search, budget fallback and
tampering. Native result pending. No learned model, promotion,
GPU or paid inference.


## Observed hosted results

Dedicated workflow 46136336969, job 140952370848, completed on
8ef364fe4b78e6c17e4ea813443ae0db7abc10de. Three regression tests passed in .54 s.
Initial run 46136307572 failed because the regression compared a cumulative
cache-hit count with one instead of the increment for that query. The
correction changes only that assertion and wraps the verifier binding
line; no change to search, proof semantics, seeds or workload. The
corrected run is the sole reported measurement.

72 fresh roots (24 native and 16 each shared, transitive and bundled),
1,008 searches. Excluded all 176 prior content IDs. 1,003 optimal
outcomes and 998 completed independently verified proofs. No incorrect
completed proof. All 504 full-cap searches were optimal and certified.
At cap 64, availability-only cover had 68/72 optimal and 64/72
certified; forced had 71/72 optimal and 70/72 certified. Every other
arm, including cheap control, achieved 72/72 optimal and certified.
The control floor is sufficient at this budget for these sampled roots,
with no universal guarantee on the inspection count.

### Cap 64

Entries: optimal/certified counts; mean masks; combined CPU ms.

| Cohort | cover | forced | closure_all | control | quotient | quotient_cached | atlas |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 23/22; 24.04; 0.763 | 24/24; 10.75; 0.840 | 24/24; 10.00; 1.458 | 24/24; 10.46; 1.545 | 24/24; 10.00; 3.056 | 24/24; 10.00; 2.654 | 24/24; 10.00; 2.881 |
| shared | 15/12; 34.63; 1.241 | 15/14; 21.94; 1.882 | 16/16; 17.13; 5.140 | 16/16; 19.31; 4.822 | 16/16; 17.13; 9.984 | 16/16; 17.13; 8.477 | 16/16; 17.13; 10.319 |
| transitive | 14/14; 17.25; 0.788 | 16/16; 9.31; 1.034 | 16/16; 7.25; 2.479 | 16/16; 8.94; 2.151 | 16/16; 7.25; 3.958 | 16/16; 7.25; 3.512 | 16/16; 7.25; 4.186 |
| bundled | 16/16; 7.63; 0.648 | 16/16; 4.44; 0.646 | 16/16; 3.06; 1.474 | 16/16; 3.75; 1.148 | 16/16; 3.06; 1.363 | 16/16; 3.06; 1.389 | 16/16; 3.06; 1.431 |

### Cap 65536

Entries: optimal/certified counts; mean masks; combined CPU ms.

| Cohort | cover | forced | closure_all | control | quotient | quotient_cached | atlas |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 24/24; 26.50; 0.934 | 24/24; 10.75; 0.823 | 24/24; 10.00; 1.467 | 24/24; 10.46; 1.541 | 24/24; 10.00; 3.054 | 24/24; 10.00; 2.603 | 24/24; 10.00; 2.860 |
| shared | 16/16; 44.81; 1.716 | 16/16; 23.19; 2.056 | 16/16; 17.13; 5.163 | 16/16; 19.31; 4.815 | 16/16; 17.13; 9.974 | 16/16; 17.13; 8.510 | 16/16; 17.13; 10.277 |
| transitive | 16/16; 22.13; 1.056 | 16/16; 9.31; 1.011 | 16/16; 7.25; 2.484 | 16/16; 8.94; 2.163 | 16/16; 7.25; 3.888 | 16/16; 7.25; 3.523 | 16/16; 7.25; 4.196 |
| bundled | 16/16; 7.63; 0.640 | 16/16; 4.44; 0.658 | 16/16; 3.06; 1.458 | 16/16; 3.75; 1.155 | 16/16; 3.06; 1.365 | 16/16; 3.06; 1.390 | 16/16; 3.06; 1.434 |

Full-cap mean additional exact-bound work and retained shallow cache
estimate (bytes; includes empty memo/atlas dicts):

| Cohort | Direct enumeration combinations | Quotient OR attempts | Cached quotient OR attempts | Cached hits | Cached bytes | Atlas work terms | Atlas hits | Atlas bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 375.54 | 1269.04 | 813.96 | 22.63 | 9012 | 889.71 | 43.13 | 4694 |
| shared | 1748.50 | 4096.31 | 2766.81 | 54.81 | 25868 | 3675.44 | 113.56 | 14117 |
| transitive | 759.38 | 1257.69 | 890.75 | 21.00 | 10390 | 1348.94 | 42.56 | 7604 |
| bundled | 403.44 | 179.00 | 178.88 | 0.06 | 5356 | 406.88 | 8.13 | 3854 |

The work counters describe different primitive operations and must not
be treated as interchangeable units. The within-world cached quotient
reduces its own OR-attempt counts by approximately 35.9% native, 32.5%
shared, 29.2% transitive, and 0.07% bundled, without changing the mean
mask counts. No bound-work fallback occurred in the measured campaign;
zero-budget fallback was exercised in regression. The bundled cohort
has many identical closure unions: atlas construction eliminates 205.94
duplicate unions on average. Yet the small number of expanded search
states limits actual reuse; quotient cache hits average only .0625.

Full-cap paired combined CPU differences, ms (95% root-bootstrap
percentile intervals; timing includes independent checker):

| Comparison | Native | Shared | Transitive | Bundled |
|---|---|---|---|---|
| control vs forced | 0.718 [0.355, 1.146] | 2.760 [1.443, 4.149] | 1.153 [0.333, 2.184] | 0.498 [0.291, 0.733] |
| quotient_cached vs quotient | -0.451 [-0.823, -0.148] | -1.464 [-2.449, -0.571] | -0.366 [-0.952, 0.044] | 0.025 [0.002, 0.047] |
| quotient_cached vs closure_all | 1.136 [0.662, 1.678] | 3.347 [1.901, 4.899] | 1.039 [0.200, 2.173] | -0.068 [-0.233, 0.111] |
| atlas vs closure_all | 1.393 [0.655, 2.271] | 5.113 [2.265, 8.270] | 1.713 [0.177, 3.875] | -0.024 [-0.210, 0.178] |

Cached quotient improves over uncached quotient in native and shared
cohorts, but loses to direct enumeration in those two and transitive.
On bundled, quotient versus direct enumeration is inconclusive.
All three exact optimized methods have identical mean mask counts
to direct enumeration on all four cohorts. The additional cheap
control floor also costs CPU versus forced in every cohort, even
though it closes the shared-control roots under cap 64. Proof
replay, setup, cache construction and lookup are not free.
The result supports the mathematical quotient and bounded proof
interface, not promotion of the cached quotient or atlas as a faster
default. Use availability/forced as the basic solver and consider the
control floor under a tighter inspection budget. Timing is exploratory:
one randomized observation per arm/root/cap on one hosted CPU,
small samples and no multiplicity correction. Cache size estimates
exclude transient allocations, proof storage and solver state.

## Disposition and next

RV23-E should investigate *when* a stronger certified relaxation is worth
invoking, before adding another learned ranker. In particular, apply the
cheap control bound lazily only to candidate nodes that remain open
after the cheap bound, and test inexpensive threshold witnesses that
prove a cutoff without computing the exact minimum. Compare against
the unchanged forced and control solvers on fresh genuinely difficult
panels. Keep full independent proof replay, separate verifier cost,
and a fixed work cap. If the overhead remains unfavorable, freeze
the exact discrimination component and return to portfolio and proof
obligation skill development. No learned-controller or checkpoint
promotion. Native actor and Baby Goblin gates remain outstanding.

Hosted run https://github.com/MoralityLabAI/DataGoblin/actions/runs/46136336969
Aggregate receipt RV23_D_DEPENDENCY_COMPACT_RECEIPT.json.
Latest general CI not yet verified when recording these results.
