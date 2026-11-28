# RV23-C: dependency-union relaxations and proof replay

Parent RV23-B 973c4225 passed both hosted workflows. CPU-only, no trained
model or private data. Use pinned Distill 7fa19c90372b13d71676e6fef13819d2a18b9be3.

## Bound and certificate

For a dependency-closed selected mask S, a deficient pair j requires d_j
additional separators. Any feasible superset T must include some d_j
unselected separators. Their transitive dependency closures are subsets of T;
therefore the minimum incremental cost of the union of exactly d_j closures
is a lower bound on the cost of a completion. The minimum is computed by
complete enumeration of the combinations, counting each combination against a
20,000-term per-solve budget. Skip the entire relaxation if it cannot fit the
remaining work allowance, preserving the cheaper bound. Max across pairs,
never sum the pair bounds (shared dependencies). Also retain the aggregate
fractional cover and cheap per-pair bounds from RV23-B. Fractional pruning
carries a rational price lambda and the checker independently validates the
dual objective lambda*D - sum_i max(0, lambda*g_i - cost_i).

Each successful complete search emits a finite proof DAG bound to the raw
world content hash. The incumbent is a feasible plan or an infeasibility
claim. The target cutoff is incumbent cost or budget+1 for infeasibility.
A leaf shows cost >= cutoff, unavailable separators, a pair/closure
bound, or a fractional dual witness. Branches enumerate every separator
closure of a deficient pair, or the union of all mandatory closures when
availability equals the deficit. The checker recomputes set-based dependency
closures, cost, separation counts, bounds and every child independently;
requires strict mask growth, no missing/orphan nodes, feasible incumbent and
matching input digest. Every feasible cheaper completion would survive at
least one branch, so closing every reachable branch proves optimality or
infeasibility. The checker is bounded to 16 experiments, 8 mechanisms,
error_bits <=2, 65536 unique nodes and 20000 closure combinations. Incomplete
search has no proof and may return a feasible incumbent or abstain. This
is an independently replayable finite Python certificate, not a Lean
formalization or authentication of externally supplied scientific inputs.

## Frozen protocol

24 fresh native frontier roots plus 12 each constructed 16-experiment
shared-control and transitive-dependency worlds, 48 roots total. Structured
worlds have six mechanisms, four nonseparating controls, 12 experiments,
one error bit and budget 18..42. Transitive cases can require other
experiments. No claim that these represent real experiments. Reconstruct
all 128 previously spent RV23-A/B robust raw root IDs and reject overlap.
Compare cover (availability only), forced (cheap pair/fractional bound
and mandatory closure), closure_one (extra exact union bound for the
branching pair), closure_all (extra bound for all deficient pairs).
Caps 64,512,65536 inspected masks, same proof-emitting engine.
All full-cap runs must match an independent exhaustive subset oracle;
native frontier optima also cross-checked against pinned upstream
reference. All returned plans independently replayed; every emitted
proof independently checked. No oracle in online search. Report
inspection, bound-combination work, search CPU, checker CPU, combined
CPU and serialized proof bytes. Proof checker work is outside the search
cap and is separately charged. One randomized arm/cap order per root,
root-paired 10000 bootstrap resamples. No tuning, no timing repetitions,
no multiplicity adjustment. Regression on 96 random small instances,
zero-term fallback, proof tampering and a shared prerequisite witness.
Numerical results pending the dedicated hosted run. No promotion.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m datagoblin.research_v2_2.closure_proof --distill-checkout "$DISTILL_CHECKOUT" --out /tmp/rv23c.json
