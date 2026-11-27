# ALG-DESCENT-03: masked Bellman residual and reuse-aware receipts (2026-11-26)

Development pilot; public-witness polynomial routing only. [Protocol](alg_descent_v3_protocol.md). Execute from repo root:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/datagoblin/research/alg_descent_v3.py > receipt_v3.json
    OMP_NUM_THREADS=1 python src/datagoblin/research/audit_alg_descent_v3.py > audit_v3.json

Python 3.13.5, torch 2.10.0+cpu, one torch thread. 60 train roots, 2,785 examples, 24 fresh IID and 24 fresh OOD roots, 912 rollouts. Three model seeds, 912 MLP parameters, 986 GRU parameters (approximately, not exactly, matched). Protocol written locally before execution, pushed afterward, so not externally timestamped pre-registration. Source SHA256 32e23b4d1f6be8670527ba6d50ca54a8b79de35dd37e7b369257e25b8fbfcc3e, verified remote blob 23f44d9c6ee8f8ad9b5397ebaab3d59b8c1b29cd. Protocol SHA256 16419d7af841ff1cecdac160042b948540001ebbbac4b5c2fbaf81850766f539. Raw local receipt SHA256 700b5d1ff4b0c6179a551ace8f6f36ee560b30580cbbfb3f2e1c16c3ae62c864, includes nondeterministic timing. Audit JSON SHA256 78e1b0ab33768d4eb49faaceca88a1ab08fc1a886413d64c4b3f37be781d7813.

All 912 rollouts closed within the work/call/receipt-byte budgets. All online policies use same exact necessary-condition mask. Corrupted candidates that survive the mask are fully checked, all attempted product operations charged. Candidate banks materialized before rollout timing; feature preprocessing, inference and bookkeeping are off-meter. Auxiliary validity target is used during training, score head alone ranks at inference. Invalid teacher target assumes optimal retry and is optimistic, not a true policy value.

| No-cache arm | IID mean work | OOD mean work | OOD closed |
|---|---:|---:|---:|
| Masked fingerprint heuristic | 217.71 | 1119.04 | 24/24 |
| MLP residual | 247.89 | 1686.82 | 72/72 |
| GRU residual | 249.24 | 1746.79 | 72/72 |
| GRU reset | 252.03 | 1764.10 | 72/72 |
| GRU reversed history | 249.24 | 1746.79 | 72/72 |

GRU vs heuristic root-averaged OOD differences +627.75 work, 12 wins and 12 losses, descriptive bootstrap interval [+210.10,+1136.17]. MLP +567.78, 13 wins and 11 losses. Pre-run gate (>=5% work improvement, >=16/24 wins, matched closure) fails for both. Reversed history produces identical work; no order-sensitive benefit established. GRU 215 exact rejections across 72 OOD runs vs heuristic 41 across 24. Mean warmed rollout time OOD heuristic 0.586 ms, MLP 1.798 ms, GRU 4.433 ms (3.886 ms measured inference including tensor construction); sequential, nonrandomized timing, no end-to-end runtime claim.

| Degree | Heuristic | MLP | GRU |
|---|---:|---:|---:|
| 24 | 521.17 | 512.44 | 524.83 |
| 32 | 915.67 | 1025.83 | 1075.33 |
| 40 | 1287.83 | 1728.89 | 1654.50 |
| 48 | 1751.50 | 3480.11 | 3732.50 |

Privileged offline no-cache DP calculated after the run averages 958.42 work. Heuristic is 16.76% above that oracle; this is restricted to the supplied candidate bank and successful no-retry/no-cache objective.

| OOD cache on heuristic | LRU | Work/byte | Reuse utility |
|---|---:|---:|---:|
| 128 logical B | 987.79 | 1000.63 | 971.38 |
| 512 logical B | 954.04 | 954.04 | 954.04 |

Reuse uses exact already-certified pending sibling counts plus observed access frequency; not a trained predictor or a probability. 128B result: 16 wins, 7 ties, 1 loss vs LRU, mean -16.42 work, descriptive bootstrap interval [-30.04,-6.96]. IID both caps tie LRU; OOD 512B ties LRU. 128B OOD reuse scans 2,302 entries, warmed average 0.339 ms vs LRU 0.312 ms. No two-budget gate pass, no speedup established. Logical byte cap excludes Python object overhead, the candidate bank, counters, pending frontier and model; it is not a RAM bound. Cache stores a trusted closure flag keyed by exact polynomial within the root, not a portable proof certificate. Utility admission compares density sums; it is a heuristic, not a knapsack optimum.

## Exact invariant and audit

For ordered positive child degrees d_i summing to d and prefixes S_i=sum_{j<i} d_j, the sequential schoolbook product cost c=sum_i (S_i+1)(d_i+1) equals (d^2-sum_i d_i^2)/2 + d+k+sum_i S_i. The first term telescopes for any complete successful uncached tree to (n^2-n)/2 when all leaves have degree one. Thus learn the structural overhead and separately price rejection risk; caching changes the accounting. The result holds only for the specified meter. The current heuristic already cancels the partition-dependent quadratic term locally, since c+sum_i d_i^2/2=d^2/2+d+k+sum_i S_i.

Independent audit: 32,132 local identities, 1,000 complete trees, 1,820 candidate products checked using independent convolution, 564 invalid, 419 eliminated by fingerprints and 145 survived. No valid action masked in this audit (also follows from evaluation homomorphism). This is an arithmetic and mask audit, not independent neural replay.

## Next experiment: ALG-DESCENT-04 (proposed)

Factor out invariant quadratic work before fitting. Separate successful overhead target, probability of rejection and paid rejection cost. Use scale-structured features and degree-relative targets; compare an exact symbolic public heuristic, residual model with no raw normalized degree and a bounded-degree model. Test counterfactual rejection and same-checkpoint history ablation; do not claim recurrence absent a matched win. For the cache, sweep 32/64/128/256/512 logical bytes plus measure RSS and bookkeeping; bound metadata and compare latency, not only work. Use fresh seeds and a published preregistration for confirmatory runs.

No native BPD/MMS integration, checkpoint publication, Lean check, factor discovery or cryptographic security inference. Raw JSON is generated locally and available separately; reproducible numerical metrics depend on software versions.
