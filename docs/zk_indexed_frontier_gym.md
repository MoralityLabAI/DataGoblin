# ZKL-15: exact indexed monomial-frontier ranking

Pinned search and identity verifier: SatsHedgeZero 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb.
This is an optimization of the ZKL-14B handcrafted frontier, with the
same static scores, dynamic overlap, seeded tie breaking and column budget.
The monomial postings lists support incremental integer intersection counts
and a versioned priority heap. When a monomial enters/leaves the residual
or enters the monotonically growing frontier, only candidates in its
postings list have their counts updated. Scores are recomputed with the
same floating-point operation order. Heap rebuilds bound stale storage.

Two fresh root blocks 15101–15108 and 15201–15208, four connected nonlinear
families, F11/BN254, budgets 24/96. Compare complete action traces and exact
certificate objects, elimination costs and lazy field-column allocations.
The original ZKL-14B routine is an additional regression reference.
Independent verifier checks every certificate. Count initial base scores,
dynamic score evaluations (including heap rebuilds), posting visits, heap
pushes, stale pops, index entries, peak heap entries and process CPU time.
The scan and heap execute in alternating order by root. CPU timing is
descriptive, not a matched wall-clock or RAM-cap study. All candidates'
symbolic support and static features are still built. The index has its
own memory overhead; no reduction in actual RSS is assumed.

No model calls, new training or production modification. The prior
ZKL-12 kernel and six generated F7/BN254 certificates passed the pinned
SatsHedgeZero workflow 46003036146. ZKL-15 per-case certificates still
require kernel replay before any stronger claim.

## Hosted result

Workflow [46008666326](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46008666326)
passed both regression tests and both 32-task blocks. All 128 paired traces,
certificates, column counts and elimination work matched. At 24 columns,
scan/indexed dynamic scores were 192624/28122 and process CPU seconds were
0.425545/0.252163. At 96 columns, scores were 567600/29997 and CPU seconds
0.877560/0.276313. The indexed arm adds 74800 postings per budget; peak heap
was 448 entries. These are descriptive hosted measurements, not an RSS claim.
