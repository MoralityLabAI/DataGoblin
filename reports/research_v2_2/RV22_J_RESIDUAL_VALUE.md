# RV22-J: residual decision values and exact shortlisting

Status: completed exploratory CPU campaign; no promotion, no paid inference.
Parent 97d8481: both hosted workflows green. RV22-J uses independently generated
Distill-compatible synthetic worlds, not native Distill teacher responses.

## Protocol

128 ordinary train roots, 32 ordinary calibration roots, and 96 each ordinary,
three-bit parity, four-bit parity evaluation roots. Split seeds 181–185, model
seeds 41–43. All serialized world identities disjoint within campaign. Earlier
RV22-I uses different generation seeds. Four-bit parity has 16 hypotheses, eight
tests and budget 9; it is a joint structure and size shift, not a pure parity
ablation. 8,004 root and first-descendant legal action examples. Exact DP supplies
training values. The actor receives only public state features, and every
selected policy is replayed over all hypotheses with per-path cost. Training and
calibration use ordinary worlds only.

Arms: direct K3 TRM, exact Q1 + nonnegative learned residual, Q1 + residual with
six branch-interaction features, Q2 + residual with the same features, and
top-two exact Q2 rescoring using the Q1 residual ranking. The six interaction
features expose exact Q2 information; they are not free. One-step, exhaustive
receding two-step, and exact DP controls share the same solver and tie handling.
The Q2 decomposition and Q2 <= optimal continuation checks passed on all 8,004
training candidates. Thresholds selected on calibration only. Top-two uses zero
stopping margin after exact rescoring. Three seed means paired by root; 10,000
percentile bootstrap replicates. No multiplicity correction.

## Results

| Arm | Ordinary utility | Parity3 utility | Parity4 utility |
|---|---:|---:|---:|
| One-step | 5.2675 | 3.9139 | 3.8652 |
| Direct TRM | 6.3619 | 6.3706 | 6.6010 |
| Q1 residual | 6.3905 | 6.4280 | 6.6391 |
| Two-step | 6.7942 | 6.6091 | 6.6352 |
| Top-two | 6.7796 | 6.6417 | 6.7983 |
| Q1 residual + pair | 6.8431 | 6.8402 | 6.8687 |
| Q2 residual + pair | 6.8484 | 6.8539 | 6.8480 |
| Exact DP | 7.0375 | 7.0967 | 7.1124 |

Paired gain relative to two-step (95% percentile CI):

| Arm | Ordinary | Parity3 | Parity4 |
|---|---|---|---|
| Q1 residual | -0.4037 [-0.5906,-0.2357] | -0.1810 [-0.4287,0.0734] | +0.0039 [-0.1304,0.1347] |
| Top-two | -0.0145 [-0.1268,0.0897] | +0.0326 [-0.1336,0.1961] | +0.1632 [0.0714,0.2580] |
| Q1 residual + pair | +0.0490 [-0.0415,0.1405] | +0.2311 [0.0535,0.4289] | +0.2335 [0.1429,0.3282] |
| Q2 residual + pair | +0.0542 [-0.0376,0.1473] | +0.2448 [0.0722,0.4399] | +0.2128 [0.1264,0.3013] |

## Compute accounting

CPU process time per complete policy tree, milliseconds, includes world
construction, features, inference and search, excludes replay and grading.
Single local run in fixed arm order; no robust timing interval. DP states count
cache misses, not total operations. Q1 and Q2 arms include cached depth-zero and
depth-one states. Neural overhead measured in the same CPU clock.

| Arm | IID ms / DP states | Parity3 ms / DP states | Parity4 ms / DP states |
|---|---|---|---|
| Two-step | 0.5245 / 187.2083 | 0.6721 / 196.9271 | 1.6764 / 528.4792 |
| Top-two | 0.5885 / 116.9236 | 0.6509 / 125.2153 | 1.3269 / 267.6701 |
| Q1 residual | 0.4024 / 52.8611 | 0.4359 / 59.1111 | 0.8979 / 121.1944 |
| Q1 residual + pair | 0.7169 / 195.4514 | 0.8065 / 205.8681 | 1.9933 / 553.5938 |
| Q2 residual + pair | 0.7155 / 195.0208 | 0.7741 / 205.8299 | 1.9347 / 554.8507 |

Parity4 shortlist reduces DP states by 49.4%, measured CPU time by 20.9%, and
regret by 34.2% compared to exhaustive two-step. IID shortlist costs more CPU.
Pair-feature arms cost more CPU than exact two-step in all three cohorts.
Feature construction is charged. Head costs 752 MACs (14 features) or 848 MACs
(20 features) per candidate, excluding nonlinearities and feature/search work.
No claim of universal speedup or a recurrence-specific advantage; no fresh
feedforward shortlist control in this pilot. Outcome-bit orientation is not
fully invariant in the pair feature head. No exact compute-matched promotion.

## Reproduce

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/residual_value.py --out /tmp/rv22j.json

Local measured source SHA-256 23289f14f567f9bc5e7e6b107120e760189738bf09f0cb67db26aae407ad3083.
Local full receipt SHA-256 76c1ac97817551aad6a94bb7ff0fc9382618bbdafcadae87daab704db6e752ec.
Published source wraps one long scope string; no intended behavioral change.
Use fresh cohorts for further inference; these three are spent. Next compare
optimized analytic search, Q1 and random shortlists, matched FF and TRM
shortlists, and confidence-gated fallback with a fixed CPU-work envelope.
