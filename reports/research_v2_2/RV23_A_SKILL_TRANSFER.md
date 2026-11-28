# RV23-A: native research-family transfer

Parent RV22-Z aa4cce37 passed both hosted workflows. New synthetic
development probe, no reference-only data admission or model/API spend.
Pinned Distill generator and exact decision assessor at 7fa19c90372b.

Three subset families share one bounded best-first search: robust
discrimination, maximin research portfolio, and conditional proof audit
selection. Family-specific public constraint/score adapters are explicit.
Dependency closure is added as one candidate; resource/group/permitted
constraints are checked. Each distinct inspected subset (including
rejected subsets) consumes one unit of the budget. Budgets 64 and 256.
No claim of matched CPU time; subset evaluation costs differ. A robust
infeasibility certificate is emitted only after exhausting reachable
search; otherwise unresolved is scored as unsuccessful. For feasible
optimal solutions the native grader accepts ties.

Train 24 fresh roots per family; 32 distinct fresh roots per evaluation
family, alternating standard/frontier, seeds 2026811000/2026812000.
No prior reference rows used. Exact local superset envelopes supply
the residual between current normalized objective and best feasible
completion. The twelve-feature ridge scorer uses public score,
progress, resource usage, subset size and interactions. Sample 64
feasible subsets per training root. Compare matched in-family fit
(24 roots) and leave-one-family-out fit (12 roots from each other
family, no target-family labels), shuffled transfer targets, analytic
priority, and random ranking. All use the same search engine and
limits. Public family-specific adapters supply substantial domain
knowledge; this tests ranking transfer, not autonomous family discovery.
Root content identity checked across train/eval; no isomorphism claim.
Native assessor/reference invoked for evaluation, not search.
Paired root bootstrap over 32 roots for transfer versus controls.
No threshold tuning. Paired causal gate evaluated separately with exact
equal-root fractions and a deliberately pooled-mean control; this is a
deterministic arithmetic gate, not another subset search family.
Adaptive evidence retains its RV22-Z result and is not silently pooled
with these tasks. Scientific prose and proposed experiments remain
ungraded.

Run pending hosted verification. Report numerical outcomes only after
receipts observed. No native TRM checkpoint, language-model training,
private actor rollouts or Baby Goblin authorization.


## Observed hosted result

Native public-transfer job 140868508551, workflow 46109135267, passed
all three regression tests and the complete native run at fd1be0117850fe21f93833c72288a27e299efaf7.
Parent RV22-Z general CI: 819 tests passed, 16 skipped, clean lint.
All 72 synthetic training roots and 128 fresh evaluation roots have
distinct raw content IDs (inventory SHA256 ea0ff83e7322613a5624690e06e2741ad4e0bdad96b1452a20da793552d635b2).
Each fit uses 1536 sampled feasible-state rows, with replacement when
a root has fewer than 64 feasible subsets. There is one fitted ridge
head per target/control, not repeated training seeds. The evaluation
uses 960 bounded subset searches and 32 paired arithmetic tasks with
two controls each. Native reference/assess calls grade finite decisions,
not full packet provenance or research prose. No oracle calls are made
inside the search ranker. All returned certificates were feasible under
the native assessor; unresolved robust cases count as unsuccessful.

Optimal certificates, including proved infeasible cases, of 32 roots:

| Family / inspected-mask cap | analytic | random | in_family | transfer | shuffled |
|---|---:|---:|---:|---:|---:|
| robust_discrimination / 64 | 16/32 | 7/32 | 16/32 | 3/32 | 1/32 |
| robust_discrimination / 256 | 20/32 | 20/32 | 17/32 | 16/32 | 13/32 |
| research_portfolio / 64 | 30/32 | 25/32 | 32/32 | 29/32 | 26/32 |
| research_portfolio / 256 | 32/32 | 31/32 | 32/32 | 32/32 | 32/32 |
| proof_obligations / 64 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| proof_obligations / 256 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

Robust discrimination: 10/32 roots are oracle-infeasible. At cap 64,
no arm exhausts its search; analytic returns 22 feasible plans, 16
optimal; held-family transfer returns only three, all optimal. The
transfer minus analytic optimum-rate delta is -0.40625 with paired
95% percentile interval [-0.59375,-0.21875]. Transfer versus its shuffled
control +0.0625 [0,0.15625]. At cap 256 transfer reaches 16/32 optimal,
analytic 20/32; the delta -0.125 [-0.28125,0]. This is a negative
transfer result, not a claim that all transfer architectures fail.
Normalized objective regret at cap 64: analytic .01351, transfer .31247.
Regret alone undercounts unresolved genuinely infeasible cases, so use
exact certificate success and answer rate jointly.

Portfolio: in-family 32/32 at cap 64, analytic 30/32, transfer 29/32.
Transfer minus analytic -0.03125 [-0.09375,0]. At cap 256 all but random
are 32/32. Proof obligations: all five arms 32/32 even at cap 64; the
task is at ceiling under this search abstraction. Exhaustion at cap
64 is 50% and cap 256 is 100% in proof obligations. These small
fixtures have limited discriminating power.

Paired causal gate: equal-root arithmetic 32/32 exact; deliberately
pooled mean 0/32 exact, with eligible-arm sets changed in 4/32 roots.
Exact certificate failure for pooled arithmetic is more frequent than
a changed decision because the native grade checks reported fractions.
This is a descriptive synthetic gate, not statistical significance.

The common hand-designed features and bounded search are real shared
components, but the public family adapters carry the semantics. Do
not infer general research transfer or native TRM benefit from this
ridge probe. Matching inspected masks does not match CPU, features,
ranking overhead, or rejected-mask cost. No timing claim. Raw-content
split checks do not exclude every structural isomorphism. All evaluated
roots are spent. No promotion or production training admission.

## Next decision

RV23-B should target the actual failure in robust discrimination:
represent capped pair-separation deficits and dependency-closure costs;
compare deficit-cover greedy and exact branch-and-bound lower bounds
at matched inspected-mask and CPU budgets. Add larger structured
proof-obligation cases only if they break the ceiling. Use a fresh
native-generated panel and keep the analytic baseline strong.
Do not fit another cross-family head until the baseline failure mode
is understood. Native actor/Jinn/Beast/Baby Goblin gates remain pending.

Hosted run https://github.com/MoralityLabAI/DataGoblin/actions/runs/46109135267
Aggregate receipt RV23_A_SKILL_TRANSFER_RECEIPT.json.
