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
