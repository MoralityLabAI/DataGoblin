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


## Observed hosted results

The dedicated public-closure-proof workflow 46135244164, job 140948966837,
completed successfully on commit b7d3686ae3abbcc8a311c061d7e6569aab0267ec. Both hosted regression tests
passed (0.34 s), covering 96 generated small instances, fallback with
zero closure-combination budget, and tampering with the example proof
binding, answer, node inventory and pruning reasons. 576 regression
full-cap solves with proof verification, plus one-mask interruption
checks. The native and structured campaign completed 576 further solves
on 48 distinct roots: 24 native frontier, 12 shared control, 12 transitive
control. Excluded all 128 raw prior RV23-A/B roots. Independent exhaustive
oracle and native frontier reference agreed; 27 roots infeasible, 21
feasible. Every returned decision was optimal in the evaluation.
All 569 emitted complete proof DAGs passed independent replay; seven
availability-only searches at cap 64 found the optimal feasible plan
but did not complete its proof within the cap. They remain uncertified.
Every arm certified all roots at caps 512 and 65536. All three bound arms
certified all 48 roots already at cap 64.

### Cap 64

Entries: optimal/certified counts; mean masks; search/check ms.

| Cohort | cover | forced | closure_one | closure_all |
|---|---:|---:|---:|---:|
| native_frontier | 24/21; 23.33; 0.772/0.355 | 24/24; 10.58; 0.985/0.315 | 24/24; 10.38; 1.056/0.311 | 24/24; 10.29; 2.017/0.318 |
| shared_controls | 12/9; 27.42; 1.200/0.327 | 12/12; 17.00; 2.013/0.489 | 12/12; 15.83; 2.138/0.483 | 12/12; 13.58; 6.353/0.570 |
| transitive_controls | 12/11; 24.08; 1.066/0.509 | 12/12; 13.83; 1.619/0.489 | 12/12; 12.17; 1.593/0.469 | 12/12; 10.83; 3.910/0.448 |

### Cap 512

Entries: optimal/certified counts; mean masks; search/check ms.

| Cohort | cover | forced | closure_one | closure_all |
|---|---:|---:|---:|---:|
| native_frontier | 24/24; 25.83; 0.816/0.537 | 24/24; 10.58; 0.987/0.313 | 24/24; 10.38; 1.048/0.310 | 24/24; 10.29; 2.047/0.317 |
| shared_controls | 12/12; 28.92; 1.233/0.681 | 12/12; 17.00; 2.023/0.494 | 12/12; 15.83; 2.166/0.498 | 12/12; 13.58; 6.323/0.551 |
| transitive_controls | 12/12; 24.25; 1.107/0.637 | 12/12; 13.83; 1.604/0.487 | 12/12; 12.17; 1.584/0.482 | 12/12; 10.83; 3.842/0.456 |

### Cap 65536

Entries: optimal/certified counts; mean masks; search/check ms.

| Cohort | cover | forced | closure_one | closure_all |
|---|---:|---:|---:|---:|
| native_frontier | 24/24; 25.83; 0.809/0.533 | 24/24; 10.58; 0.975/0.311 | 24/24; 10.38; 1.051/0.307 | 24/24; 10.29; 2.015/0.313 |
| shared_controls | 12/12; 28.92; 1.221/0.678 | 12/12; 17.00; 2.019/0.501 | 12/12; 15.83; 2.159/0.487 | 12/12; 13.58; 6.329/0.543 |
| transitive_controls | 12/12; 24.25; 1.083/0.631 | 12/12; 13.83; 1.617/0.485 | 12/12; 12.17; 1.594/0.464 | 12/12; 10.83; 3.865/0.476 |

At full cap, mean all-pairs closure combination counts are 390.58,
1588.17, and 841.58 on native, shared, and transitive cohorts.
Single-pair counts are 22.38, 41.00, and 27.42. Mean serialized proof
bytes for forced/closure_one/closure_all: native 355.21/351.92/350.92,
shared 468.42/453.33/419.00, transitive 425.67/399.58/377.50.
The full-cap cover proof means are 597.29/668.67/594.17 bytes. Proof
sizes exclude original world bytes and interpreter state. Aggregate
receipt stores counts/means/paired intervals, not all per-root proof DAGs.

Full-cap paired CPU differences (search plus independent check), ms:

| Comparison vs forced | Native | Shared | Transitive |
|---|---|---|---|
| closure_one | +0.072 [0.023,0.122] | +0.127 [0.002,0.258] | -0.044 [-0.213,0.077] |
| closure_all | +1.041 [0.545,1.632] | +4.352 [1.721,7.584] | +2.239 [1.119,3.418] |

All-pairs lowers inspected masks by 3.417 [-7.250,-0.417] signed
delta on shared and 3 [-5.5,-1] on transitive controls, but raises total
CPU significantly on these small cohorts. Single-pair transitive
inspection delta -1.667 [-3.417,-0.250] with combined CPU interval
crossing zero. Availability-only branching has the lowest mean combined
full-cap CPU on the two structured cohorts, but seven incomplete
certifications at cap 64 overall. Use the cheap forced-bound solver
as the tighter-budget reference. The stronger mathematical bound is
sound, yet its cost did not repay itself here. Do not promote the
all-pairs method or infer a speedup from mask counts. Timing is a
single randomized observation per root/arm/cap and the sample is small;
no broad hardware or statistical-generalization claim.

Source blob 4483a2ff9491074656431e64642d6a3f165bf48d.
Measured regression blob 00adcdc0ad5c52077a1ac5620bbf492c1279dca0.
Hosted run: https://github.com/MoralityLabAI/DataGoblin/actions/runs/46135244164
Aggregate receipt RV23_C_CLOSURE_PROOF_RECEIPT.json. Example fixtures
RV23_C_EXAMPLE_PROOFS.json reproduce the shared-control regression with
budget 8 (infeasible, root union bound 9) and budget 9 (optimal plan
cost 9, mandatory closure). An additional regression binds these
published fixtures to the solver and checker; its hosted CI is pending.

## Next: RV23-D

Reduce the cost of union relaxations by quotienting candidate closures
by their residual dependency masks. Cache or reuse exact closure-union
costs where the selected state induces equivalent residual sets, and
compare a cheap control-cost lower bound that avoids enumerating all
combinations. Charge lookup, construction, proof verification and
memory. Use fresh larger structured panels only after validating the
bound on exhaustive small instances. Preserve availability-only and
forced-bound controls. No learned ranking or checkpoint promotion
is justified by this campaign.
