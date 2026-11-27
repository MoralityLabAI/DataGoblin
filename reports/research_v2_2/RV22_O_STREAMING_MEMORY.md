# RV22-O: bounded streaming research-memory controller

Local CPU experiment completed. No GPU, paid inference, external model calls,
or controller promotion.

## Protocol

Use the RV22-N integer mask/budget DP, exact canonical byte key, independent
Bellman checker, and policy replay. A 64-slot direct-mapped sketch stores a
64-bit BLAKE2b digest hint, 32-bit last-observation index, saturated repeat count
and padding per 16-byte entry. An 8-byte header holds a Q8 exponentially weighted
recent-repeat rate (alpha 1/8). A six-coefficient additive ridge head predicts the
number of exact-key repeats in the next 12 queries from bias, repeat, count,
gap, EWMA and repeat-times-EWMA. Digest collisions may change admission only;
cache hits require the full canonical byte key and replay. No trust downgrade:
checked arms verify every newly admitted entry.

The accounted cap is 8192 bytes of canonical keys plus serialized policies,
with a fixed 1080-byte reserve for the 1032-byte sketch and six float64
coefficients. Adaptive arms therefore have 7112 cache-payload bytes.
Matched LRU arms reserve the same allowance without allocating the sketch;
full LRU arms have the whole payload budget. The Python implementation uses
ordinary objects for coefficients and LRU storage. Report a separate shallow
retained-object estimate, excluding temporary DP allocations and allocator/RSS.
This is a strict accounting cap, not an RSS bound.

Train on 48 streams of 72 queries, burst/mixed/churn only, 2880 labeled
observations and 1216 unique canonical keys. Fit a same-feature shuffled-target
head. Fixed decision thresholds: .25 predicted repeats for trusted admission,
1.5 for checked admission. These are heuristic thresholds rather than
calibrated cost ratios. The analytic checked gate admits a recent repeat with
at least two prior consecutive sketch observations or sufficiently high EWMA
and gap <= 4; trusted gate admits a recent repeat or EWMA >= .50.
Second-observation admission uses the same sketch. All caches use LRU eviction.
Every non-cold arm still pays for canonicalization; bypass avoids admission,
not the key lookup itself.

Smoke campaign: one stream per cohort at separate seed range. Main evaluation
uses EVAL_BASE=2026150000, 12 streams per cohort, 64 queries per stream, six
cohorts, two randomized timing repeats. Phase switches burst to churn; recovery
switches mostly novel queries to bursts; scan interleaves hot roots and novel
queries. These last three workload patterns are absent from training. Exact
canonical-key inventory checks establish zero intersections between train and
each evaluation cohort and between cohorts; not a complete structural
isomorphism audit. Evaluation cohort unique keys 96,117,627,378,391,492.
Each timing repeat starts with empty memory. All 110592 arm-query executions
agree exactly on utility numerator and denominator (9216 per arm, 4608 query
occurrences before repeats). Paired percentile bootstrap uses 10000 draws over
12 streams per cohort after averaging two repetitions within stream. Timing
includes canonicalization, sketch, inference, solver, checker and replay,
excludes offline fitting, evaluator and retained-size measurement.
No multiple-comparison correction or hardware transfer claim.

## Results

Mean CPU milliseconds per 64-query stream:

| Arm | Burst | Mixed | Churn | Phase | Recovery | Scan |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 66.669 | 61.183 | 66.095 | 73.094 | 65.579 | 72.384 |
| Trusted LRU | 14.158 | 14.156 | 66.806 | 42.694 | 39.302 | 46.928 |
| Trusted matched LRU | 13.096 | 14.400 | 66.225 | 50.617 | 39.655 | 47.533 |
| Trusted second | 22.10 | 26.97 | 72.24 | 47.28 | 45.80 | 49.91 |
| Trusted analytic | 14.582 | 20.794 | 71.466 | 41.285 | 43.319 | 49.572 |
| Trusted learned | 14.888 | 15.238 | 66.750 | 42.023 | 41.332 | 47.677 |
| Checked LRU | 23.645 | 26.213 | 139.753 | 86.968 | 77.863 | 95.431 |
| Checked matched LRU | 23.996 | 30.279 | 136.192 | 88.237 | 80.688 | 94.904 |
| Checked second | 31.776 | 37.126 | 72.431 | 55.426 | 53.567 | 51.377 |
| Checked analytic | 32.545 | 42.096 | 71.789 | 50.253 | 52.735 | 54.083 |
| Checked learned | 27.441 | 31.768 | 72.633 | 54.989 | 51.572 | 72.042 |
| Checked shuffled | 23.180 | 26.457 | 132.163 | 83.850 | 78.558 | 94.088 |

Checked learned minus matched checked LRU, paired mean [95% CI] ms:

| Cohort | Delta [95% CI] |
|---|---|
| Burst | +3.444 [2.053,5.027] |
| Mixed | +1.489 [-6.107,6.257] |
| Churn | -63.559 [-70.623,-55.903] |
| Phase | -33.248 [-43.018,-25.348] |
| Recovery | -29.116 [-32.589,-25.601] |
| Scan | -22.862 [-28.691,-18.078] |

Learned checked admission eliminates churn admissions (58.667 to zero),
and estimated peak retained objects fall from 10.01 KiB (matched LRU) to
1.563 KiB. Nevertheless CPU remains +6.538 [3.148,10.868] ms above cold.
Checked analytic and second-observation admission also lose to cold there.

The checked learned head beats its shuffled target control on churn, phase,
recovery and scan; it loses on burst and mixed. It loses to checked analytic
on phase (+4.736 [3.326,6.173]) and scan (+17.959 [13.452,21.258]).
Recovery learned versus analytic is inconclusive (-1.163 [-3.440,.926]).
Scan's cheap second-observation gate checks 3.0 times per stream compared
with 19.083 for learned admission, and saves CPU versus cold
(-21.007 [-45.479,-7.950]). The scan timing intervals are wide.

No universal learned advantage. Trusted LRU remains a strong cheap default;
when independent checking is required, checked LRU is attractive on stable
reuse and compact analytic/second-observation gates help on shifts.
No controller promotion: a future policy needs actual key-generation bypass
and a prospectively defined workload switch rather than selecting a best
arm retrospectively.

## Verification

Two local pytest functions passed. Independent index/used-test recursion
matched DP and replay on 24 worlds (12 base/variant pairs). Forced identical
64-bit tags for distinct worlds did not authorize a false cache hit.
All 12 arms activated in regression, checked entries counted, capacity enforced,
and zero cache allowance safely declined admissions.

RV22-N hosted tests: 792 passed / 15 skipped, dedicated Research Verifiers
workflow green; general CI failed on one unused unpacked variable (RUF059).
Renamed it to _last in RV22-O. Latest hosted CI remains unverified.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/streaming_memory.py --out /tmp/rv22o.json

Measured source Git blob 1288385b4097494fcef1dbc35512401a7757b301.
Test Git blob 3d744b949be4abdfa80e55f94df00c2eafd839c7.
Measured source SHA-256 9f0c2713ad9e357bdcad6c6df0b01055f2a29fbbfa843c2634cf69f4212aedfa.
Receipt SHA-256 68fbf315ea588811aa7d3a7a5b9bc659f473f2feedf52ec3622ec98b0e41a23f.

## Next

RV22-P: cost-sensitive cold/cache routing with a cheap, sampled reuse
monitor that can bypass full key construction on cold streams. Compare
static cold, full LRU, analytic switch and learned switch, separately for
trusted and checked admission. Charge sampled canonical probes, detection
delay and misses after workload changes. Freeze thresholds on development
streams; use fresh roots and previously unseen workload patterns.
