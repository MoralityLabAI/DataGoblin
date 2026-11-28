# RV22-N: economic root-memory admission

Completed local synthetic CPU study. No paid inference, GPU usage, or controller
promotion. Parent RV22-M: both hosted workflows green.

## Protocol and audit

Self-contained implementation of the prior integer mask/budget exact solver,
canonical root equivalence (including permutation, weight scaling and duplicate
splitting), independent bottom-up Bellman checker, and original-hypothesis replay.
This script is scoped to generated valid finite deterministic-test inputs. It is
not an external proof importer or a replacement for the native checked-cache API.

Train on 36 streams of 60 queries (2,160 timings, 1,728 supervised examples
with a complete 12-query future horizon). Fit an 11-coefficient ridge head to
the observed sum of positive cold-solve minus decode/replay times of future
canonical repeats, conditional on public current size, history, and measured
miss cost. Fit an equal-feature shuffled-target control. No test utility or future
identity is supplied at inference. Training median hit time .025971 ms;
median check/solve ratio 1.238125. Mean target 1.74616 ms. Training costs excluded
from inference campaign timing; no learned model amortization claim.

Compare cold DP, trusted LRU/analytic/learned root cache, checked LRU, checked
second-observation, checked analytic/learned/shuffled admission. Use the same
8192-byte serialized root key+policy cap for all arms. History cap 128 entries;
full history keys and Python metadata are outside that cap. Report a shallow
retained object size estimate including history (can double-count shared
references; excludes temporary allocation and allocator/RSS). This estimate is
computed outside timed regions. Economic eviction selects low remaining value
per byte, then admits if estimated gross future savings exceed verification
estimate, heuristic .01 ms admission cost and estimated foregone victim savings.
Victim value decays linearly with age. These are approximate decision scores,
not an exact optimum for replacement.

Initial development run revealed overlapping seed ranges between training and
evaluation. Discarded its results from the primary comparison. Replaced test
seeds with 2026128000+100*cohort+ordinal; training seeds remain
2026120100+100*cohort+ordinal. Final audit found 804 training canonical keys,
96 burst / 148 mixed / 667 churn / 381 phase evaluation keys, and zero
intersections across all five sets. No claim of complete isomorphism auditing.
The first-run workload families had already been explored; the rerun is
fresh-root exploratory evidence, not a preregistered untouched family study.

Final evaluation: 16 streams per cohort, 48 queries per stream, four cohorts,
3,072 queries per arm and 27,648 total arm-query executions. Phase cohort switches
from burst to churn halfway through; no phase streams in training. All nine
arms returned the same exact numerator/denominator pairs. Arm order shuffled
within each stream. CPU process clock includes key generation, solving, replay,
prediction, eviction and verification; excludes memory instrumentation and
evaluation grading. Paired stream bootstrap 10,000 replicates. No correction
for multiplicity; timing is local and noise-sensitive. The trusted LRU churn
interval is particularly wide. Separate used-mask recursive oracle audit:
96 exact comparisons, 61,116 checked Bellman edges, 96 corrupted roots rejected.

## Results

Mean CPU milliseconds per 48-query trajectory, corrected evaluation:

| Arm | Burst | Mixed | Churn | Phase |
|---|---:|---:|---:|---:|
| Cold DP | 50.130 | 46.907 | 48.932 | 46.335 |
| Trusted LRU | 10.339 | 13.131 | 68.540 | 28.975 |
| Trusted analytic | 9.530 | 13.744 | 52.297 | 29.195 |
| Trusted learned | 10.516 | 14.220 | 50.949 | 30.626 |
| Checked LRU | 16.605 | 23.870 | 103.184 | 56.250 |
| Checked second | 24.328 | 29.531 | 57.406 | 39.192 |
| Checked analytic | 33.224 | 34.621 | 55.775 | 38.256 |
| Checked learned | 21.891 | 26.425 | 69.412 | 45.579 |
| Checked shuffled | 31.663 | 32.698 | 73.812 | 49.985 |

Checked learned minus checked LRU, paired mean delta [95% CI] ms:

- Burst +5.286 [4.188, 6.451]
- Mixed +2.555 [0.382, 4.698]
- Churn -33.773 [-36.906, -30.468]
- Phase -10.670 [-15.931, -2.787]

Checked analytic beats checked learned on churn and phase means; it is slower
on burst and mixed. Against cold, checked analytic churn remains +6.843 ms
[4.899, 9.038]; phase -8.079 ms [-10.360, -5.821].
Checked LRU is fastest checked mean on burst and mixed. Learned admission
reduces churn checks from 44.375 to 14.125 per stream, but misses the
cold reference by +20.480 [17.903, 23.144] ms. The shuffled target control
also reduces admission; no direct paired learned-vs-shuffled interval is
reported, so no learned-specific speed claim. Trusted LRU saves CPU in burst,
mixed and phase; churn has rare reuse and noisy timings. All candidate
policies preserve exact utility.

On churn the trusted LRU mean peak serialized payload is 8,154.8 bytes and
mean peak estimated retained Python objects is 14,073.1 bytes. Trusted learned
has the same mean payload and 42,887.3 bytes including history. The economic
metadata is itself a cost. Checked analytic rejects every churn admission,
but still retains history (29,443.7 estimated bytes). Memory cap is not a
resident RAM guarantee. Need a compact bounded sketch or explicit hard
accounting for history before any memory promotion.

## Reproduce

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/economic_memory.py --out /tmp/rv22n.json

Measured source SHA-256 9e5dbf02adfe93f1212770043b3ba35f80372287092b6795b108926a80e08ac9.
Published source adds one explanatory comment, Git blob ce3bdc44c8d4cd90b5b590505290b85bc6dd9079.
Receipt SHA-256 9897c78f24781d953cf3df002c0d3e98b23b49b39238948171e69220d66c6cce.
Audit SHA-256 9e916ff4fe529f50e4a2213bb06a60fddfdf8369f79ea82200b506a55eba72c2.
CPU timing and fitted coefficients vary across runs.

Next: a compact streaming reuse detector (bounded digest history, recency
and hit-rate EWMA) that routes between cold, trusted LRU and checked-root
admission. Compare static and online regimes at matched resident-memory
budgets. Use fresh workload-family shift and deterministic logical-work
receipts alongside repeated randomized CPU timing. No promotion.
