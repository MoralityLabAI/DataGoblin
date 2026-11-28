# RV23-B: exact dependency-aware robust experimental design

Parent RV23-A: 822 tests passed / 16 skipped; dedicated verifier CI
green, general CI failed one RUF021 parenthesization warning, fixed here.
Use native Distill robust_discrimination generator and assessor pinned
to 7fa19c90372b13d71676e6fef13819d2a18b9be3. No learned fit, reference
data admission, model API, GPU, or promotion.

## Algorithm and proof

All experiments have positive cost. Selecting an experiment includes
its transitive dependency closure. A deficit D_j=max(0,2e+1-count_j)
is computed for each unordered mechanism pair. If fewer unselected
separating experiments remain than D_j, no completion exists. If their
number equals D_j, every remaining separator and its closure is mandatory
in every feasible completion. Otherwise, branch on the deficient pair
with smallest separator slack (then candidate count, then pair index).
Every feasible completion includes at least one separator for this pair.
Hence branching over their closures, or over the union of mandatory
closures, preserves a path to every potentially improving completion.
Supersets of complete panels cannot improve under positive costs.

Relax dependencies when computing lower bounds. Per-pair bound is the
maximum, over pairs, of the sum of their D_j cheapest remaining
separator costs. Aggregate fractional relaxation: D=sum_j D_j;
each remaining item contributes at most its number g of currently
deficient pairs separated. Any feasible completion has total g>=D.
Minimize cost with fractional item quantities in [0,1], greedily sorted
by exact cost/gain. Round the minimum up since original costs are
integers. Both bounds are admissible; their maximum plus current cost
cannot exceed the cheapest feasible completion. Prune when this value
cannot improve the incumbent or fit budget. Exact full-key mask
deduplication is safe because all future costs/constraints depend on
the selected mask. A cap interruption leaves proof status false.
An infeasibility answer is emitted only after a complete certified
search; an uncompleted greedy search abstains.

## Frozen evaluation

Six arms: RV23-A legacy analytic, dependency-closure deficit-gain/cost
greedy, targeted pair branching with availability bound, targeted
branching with cost bounds, forced closure propagation with bounds,
and greedy-warmed forced search. Count every newly inspected mask
including rejected over-budget masks. Bound evaluations and generated
edges reported separately. Caps 64/256/4096. No speculative full subset
oracle calls inside online search. CPU process time includes solver
setup, search, bound scans and certificate construction, excludes
native oracle grading. One randomized arm/cap order per root. Native
assessor accepts optimal ties. Certified implies optimal according to
native grader; all exact branching arms must certify at full cap.

24 fresh roots per difficulty (foundation/standard/frontier), 72 roots,
1296 bounded searches. Raw native content IDs disjoint from reconstructed
RV23-A robust training/evaluation roots and between new cohorts; not
a general isomorphism audit. Paired bootstrap 10000 root resamples
within difficulty, exploratory intervals, no multiplicity correction.
The logical inspection budget is common but not a machine instruction
budget; CPU is separately measured, one timing observation per root.
Do not count unresolved oracle-infeasible roots as successes.
Only returned finite-model certificates are graded, not scientific
interpretation or research design prose. Native result pending.

Regression independently enumerates subsets on 48 small positive-cost
cases with dependencies, checking lower bounds and forced closures on
every feasible-superset state, exact full-budget results, bounded
abstention and a targeted forced dependency example.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m datagoblin.research_v2_2.deficit_cover --distill-checkout "$DISTILL_CHECKOUT" --out /tmp/rv23b.json


## Observed hosted results

Hosted workflow 46123880484, public-deficit-cover job 140914042038,
completed successfully on commit f9a01f116877164e9f050b91bcc8269a4fe75903. Both hosted regression
tests passed, including the independent 48-case exhaustive bound audit.
The actual pinned native generator and assessor completed 1296 searches
across 72 fresh roots (24 per difficulty) and 6 arms x 3 limits.
All 56 prior RV23-A robust train/eval roots were reconstructed and
excluded by raw content ID. No overlapping new content IDs. Native
infeasible counts foundation 1, standard 0, frontier 11. Every
returned plan is feasible, every emitted infeasibility answer is
correct, and every internal certified-optimal status agrees with the
native assessor. Some feasible answers are suboptimal and explicitly
uncertified. This is a search-completion status with an independent
native re-solve audit, not a separately portable optimality proof
transcript or Lean theorem.

### Cap 64

Each entry is optimal/certified counts of 24; mean masks; mean CPU ms.

| Difficulty | legacy | greedy | cover | bound | forced | warm_forced |
|---|---:|---:|---:|---:|---:|---:|
| foundation | 23/1; 63.83; 0.743 | 21/0; 15.50; 0.118 | 24/24; 8.46; 0.087 | 24/24; 6.54; 0.127 | 24/24; 6.54; 0.127 | 24/24; 16.04; 0.178 |
| standard | 21/0; 64.00; 0.774 | 22/0; 24.42; 0.226 | 24/24; 14.25; 0.167 | 24/24; 10.04; 0.262 | 24/24; 10.04; 0.260 | 24/24; 27.58; 0.362 |
| frontier | 10/0; 64.00; 0.899 | 11/0; 58.38; 0.709 | 23/22; 15.04; 0.319 | 24/24; 12.08; 0.603 | 24/24; 7.71; 0.356 | 23/21; 60.04; 0.849 |

### Cap 256

Each entry is optimal/certified counts of 24; mean masks; mean CPU ms.

| Difficulty | legacy | greedy | cover | bound | forced | warm_forced |
|---|---:|---:|---:|---:|---:|---:|
| foundation | 24/24; 89.42; 0.887 | 21/0; 15.50; 0.116 | 24/24; 8.46; 0.086 | 24/24; 6.54; 0.128 | 24/24; 6.54; 0.127 | 24/24; 16.04; 0.178 |
| standard | 23/6; 250.17; 2.972 | 22/0; 24.42; 0.229 | 24/24; 14.25; 0.165 | 24/24; 10.04; 0.274 | 24/24; 10.04; 0.264 | 24/24; 27.58; 0.363 |
| frontier | 13/0; 256.00; 4.310 | 11/0; 58.38; 0.705 | 24/24; 17.67; 0.360 | 24/24; 12.08; 0.598 | 24/24; 7.71; 0.361 | 24/24; 62.63; 0.941 |

### Cap 4096

Each entry is optimal/certified counts of 24; mean masks; mean CPU ms.

| Difficulty | legacy | greedy | cover | bound | forced | warm_forced |
|---|---:|---:|---:|---:|---:|---:|
| foundation | 24/24; 89.42; 0.869 | 21/0; 15.50; 0.117 | 24/24; 8.46; 0.087 | 24/24; 6.54; 0.128 | 24/24; 6.54; 0.128 | 24/24; 16.04; 0.178 |
| standard | 24/24; 294.33; 3.608 | 22/0; 24.42; 0.225 | 24/24; 14.25; 0.166 | 24/24; 10.04; 0.259 | 24/24; 10.04; 0.263 | 24/24; 27.58; 0.368 |
| frontier | 24/24; 1208.21; 20.597 | 11/0; 58.38; 0.707 | 24/24; 17.67; 0.356 | 24/24; 12.08; 0.602 | 24/24; 7.71; 0.358 | 24/24; 62.63; 0.947 |

At cap 64, forced and bound each certify 72/72 roots (60 optimal plans
and 12 infeasibility decisions), legacy finds 54/72 optimal answers
and certifies only 1, cover finds 71/72 and certifies 70/72. Greedy finds
54/72 optimal plans and never claims exactness; warm forced finds 71/72
optimal answers and certifies 69/72. Greedy warming can consume the
limited inspection budget. Frontier forced minus legacy optimum-rate
delta +0.583333 [0.375,0.791667] (root-paired percentile interval).
At cap 4096 all four exact branching variants and legacy certify all
72 roots; forced frontier mean 7.7083 vs legacy 1208.2083 masks
(99.36% fewer), 0.358448 vs 20.596722 ms (~57.46x). Paired CPU
delta -20.238274 [-21.743833,-18.769403] ms. This is the comparison
with the previous general-purpose analytic implementation, not an
optimal compiled multicover baseline. Frontier forced vs bound:
4.375 fewer masks and -.243916 [-.348072,-.142600] ms.
Cover has faster mean CPU on foundation/standard than bound/forced
despite more masks; on frontier full budget, forced and cover mean
CPU is essentially equal (+.002160 [-.056629,.089489] ms).
Bounds need not repay their computation. The native cases are small;
no claim of general multicover speedup, matched CPU-budget dominance,
learned skill promotion, or general research competence. Timing is a
single randomized process-CPU observation per arm/root, exploratory.
Inspection caps exclude lower-bound primitive work; separate counters
and total CPU cost are published in the receipt.

Measured source blob 8c63b455313438f2aba3521a3d303ac36eb1bf24.
Test blob d5e727ec6e0b228b0a00012cb033fc85fc9b7c33.
All evaluation roots are spent.

## Next: RV23-C

For harder cases, compare the relaxed per-pair bound against an exact
single-pair dependency-closure relaxation (minimize the union cost
of D separating choices). This may capture mandatory control costs
without charging shared controls twice. Add structured larger panels
and explicitly meter bound computation, then export independently
replayable optimality/infeasibility certificates. Keep the cheap
pair-branching and forced solver as controls. Do not retrain the
cross-family head unless it solves a demonstrated residual problem.

Hosted run: https://github.com/MoralityLabAI/DataGoblin/actions/runs/46123880484
Receipt: RV23_B_DEFICIT_COVER_RECEIPT.json.

The receipt commit also flattens one nested full-budget assertion conditional for lint; no intended semantic change.
