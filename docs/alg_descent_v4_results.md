# ALG-DESCENT-04: potential-normalized routing and cache sweep (2026-11-26)

Development experiment on synthetic public-witness polynomial certificates. This is not factor discovery or a discrete-log algorithm. Protocol registered before execution at [0db0db6](https://github.com/MoralityLabAI/DataGoblin/commit/0db0db6552adf58656e7c37e2c0a2ef4d62884fc); publicly specified roots are not a sealed holdout.

## Reproduction and scope

From repo root, on Linux with PyTorch:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/datagoblin/research/alg_descent_v4.py > receipt_v4.json
    OMP_NUM_THREADS=1 python src/datagoblin/research/audit_alg_descent_v4.py receipt_v4.json > audit_v4.json

Python 3.13.5, torch 2.10.0+cpu, one torch thread. 80 training roots, 13,239 candidate/history examples, 40 empirical prior buckets, 24 IID and 32 OOD fresh roots. Three model seeds. MLP 1,262 parameters (both variants); GRU 1,082. 1,400 primary rollouts plus 56 cache-worker repetitions of the zero-cache baseline, which reproduced work exactly. The only two closure failures are structural-only IID roots 120008 and 120014. Source SHA256 759b5884a8d6b9f69cb0a71eff718260cdad40918367675f1712d32367e0502f, git blob c546eb1ea59369d00d0001df80df74019728f349; audit source SHA256 4d2ec1cca7a820cf6448f14dae4c0d4a9deb965de7efbc7a503458ae1a2f7a69, git blob 7753f6e89b4510c1dcbcc8fa21024ce78338b1a3. Both match remote blob hashes. Local receipt SHA256 7469c050b64fa42823a50890bec1c46504b8200d536e329aadc7204dd9e8ee99, audit JSON SHA256 377e4f5f27e1090c9146dc57ad35373376f346a579046a3d91a14c75edb6cac7. Timing and RSS make the receipt hash run-specific. Weights and raw receipt not committed; seed-based reproduction supplied.

The structural bound is exact over unrestricted positive ordered degree partitions with arities 2,3,4,8, but not necessarily achievable with a given public certificate bank. The symbolic prior estimates eligibility-conditioned validity from training counts (Laplace smoothing, clipping to [.05,.995]), keyed by characteristic, arity and balanced-degree pattern. The same probabilities are available to neural arms. Learned successful overhead excess is nonnegative, normalized by degree; risk is a logit residual. The scale-free arm removes explicit log degree, but remaining descriptors may still reveal degree. Model sizes are not exactly matched; reset uses the same GRU checkpoint. Model history is local to a node, last two failed eligible proposals.

For independent candidates with fixed probabilities p_i>0 and fixed acceptance-contingent downstream costs V_i, the optimal fixed retry order is increasing C_i/p_i+V_i. Correlation and posterior changes make this an approximation in the gym. After subtracting the fixed quadratic potential, the score is structural(a)+estimated_excess(a)+C_i*(1/p_i-1). Exact verification remains mandatory. Every attempted product operation charged; fingerprints, construction, inference, score computation, terminal checks, lookups, bookkeeping are off synthetic work meter. Candidate banks precomputed (2.506 s total over evaluation roots). Report warmed rollout time separately; source times score computation within rollouts. No full end-to-end timing claim.

## Results

| No-cache arm | IID closed | IID mean work | OOD closed | OOD mean work | OOD mean rejected work | OOD warmed ms |
|---|---:|---:|---:|---:|---:|---:|
| v3 heuristic | 24/24 | 242.00 | 32/32 | 1157.84 | 134.47 | 0.518 |
| structural | 22/24 | 287.62 | 32/32 | 1554.28 | 586.66 | 0.529 |
| structural+risk | 24/24 | 214.58 | 32/32 | 1042.12 | 44.28 | 0.724 |
| scale-free MLP | 72/72 | 223.21 | 96/96 | 1031.61 | 34.58 | 3.461 |
| scale-aware MLP | 72/72 | 220.71 | 96/96 | 1031.07 | 33.79 | 3.410 |
| GRU | 72/72 | 212.00 | 96/96 | 1031.17 | 33.80 | 6.082 |
| reset GRU | 72/72 | 224.50 | 96/96 | 1032.42 | 35.46 | 6.103 |

All-run spent work is secondary to verified closure. Risk-aware symbolic baseline improves by ~10% vs v3 heuristic; v3 heuristic minus risk mean +115.72 on 32 OOD roots, risk wins 25/32. Descriptive paired bootstrap interval for that difference [+48.25,+202.31]. Scale-aware MLP mean -11.05 vs risk (13 wins,12 ties,7 losses; interval [-25.86,+0.48]); GRU -10.96 (15 wins,10 ties,7 losses; interval [-25.90,+0.53]); scale-free -10.51 (13 wins,11 ties,8 losses; interval [-25.34,+1.00]). Root-averaged seeds, no treatment of 96 model runs as independent roots. The neural arms fail registered >=5% and >=20/32 wins gate. GRU minus same-checkpoint reset mean -1.25; 2 wins,28 ties,2 losses. Recurrent contribution not established. Neural warmed time much higher; no speedup.

| Degree | v3 heuristic | symbolic+risk | scale-aware MLP | GRU |
|---:|---:|---:|---:|---:|
| 24 | 495.38 | 475.12 | 472.62 | 474.83 |
| 32 | 875.12 | 815.62 | 817.50 | 816.88 |
| 40 | 1290.12 | 1223.12 | 1185.62 | 1180.42 |
| 48 | 1970.75 | 1654.62 | 1648.54 | 1652.54 |

The severe v3 large-degree failure is absent on these fresh roots. The across-version comparison changes roots and policy design; the registered comparison against the v3 heuristic within v4 is the evidence for improved work.

## Cache sweep on structural+risk

| Logical bytes | LRU work | Pending utility work | Pending wins/ties/losses | LRU warmed ms | Pending warmed ms |
|---:|---:|---:|---:|---:|---:|
| 32 | 903.16 | 891.94 | 5/27/0 | 0.779 | 0.785 |
| 64 | 879.62 | 866.84 | 11/21/0 | 0.826 | 1.115 |
| 128 | 828.56 | 824.97 | 3/29/0 | 0.740 | 0.788 |
| 256 | 811.19 | 811.19 | 0/32/0 | 0.733 | 0.665 |
| 512 | 809.69 | 809.69 | 0/32/0 | 0.809 | 0.754 |

All cache arms close all roots. Pending utility uses only already-accepted future sibling multiplicities times observed saved work (minus hit charge), divided by size for victim ordering; admits if total estimated value covers victims. This is a heuristic, not an optimum or a trained probability. No global history; remove zero pending counters. LRU control tracks the same pending counts for instrumentation. Maximum OOD nonzero pending keys: 5. Cache work improvements at 32/64/128 bytes have paired bootstrap intervals [-22.47,-3.06], [-24.50,-4.75], [-9.375,0], respectively. Both work and time gate fails; isolated workers are sequential, not timing-randomized across capacities, and score/instrumentation overhead can dominate. No-cache isolated worker OOD mean 1.114 ms versus 0.724 ms in parent, illustrating environment/timing effects. Avoid cross-process latency claims on such small absolute differences.

Logical byte charge len(polynomial)+16 is a simulation. Conservative object-size estimate (including dictionary, key tuple/coefficients and value tuple/integers, may overcount shared objects) peaked at 1,004/2,012/3,724/7,244/12,692 bytes for 32/64/128/256/512 logical cap. Actual peaks 29/64/128/256/511 logical bytes. Forked worker sampled RSS peaks 445.18..445.69 MiB; inherited precomputed banks and runtime dominate, shared pages count in RSS. No PSS/private memory measurement, exact physical cap, or allocation reduction demonstrated. Pending keys, stack, candidate bank and model remain off logical cap.

## Independent checks

4,459 compositions enumerated through degree 14 via independent schoolbook expansion and exhaustive partition recurrence. 300 random exact rational priority instances, 11,594 permutations, all consistent with adjacent-exchange rule. 1,400 receipt rows checked against budget; all 838 closed uncached rows satisfy quadratic decomposition; 34 summary cells recomputed. No independent model replay or full certificate regeneration by this audit. The actual experiment checks each accepted product.

## Decision and next proposal

FAIL learned-superiority and combined cache work/time gates. Positive symbolic improvement, small conditional cache work advantage at tight budgets, no neural or native BPD/MMS superiority established. ALG-DESCENT-05 should put the deterministic router and cache on a native packed representation, compare matched physical allocation capacities, and record end-to-end latency including bank generation and lookup. Also test whether calibrating the risk estimate on corruption regimes held out by family improves generalization; a model should compete with the symbolic prior on the same public features and preprocessing budget. For BPD, target a measurable risk-prediction improvement first (calibration on genuinely new generator families), then route under equal cost. Avoid another neural scaling run on this narrow generator before the native baseline. No Lean verification performed.
