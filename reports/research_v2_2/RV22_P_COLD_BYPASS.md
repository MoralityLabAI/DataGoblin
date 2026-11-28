# RV22-P: sampled cold bypass

Completed local synthetic CPU campaign. Parent RV22-O passed both hosted
workflows. No external model calls, paid compute, GPU use, or promotion.

## Method

Self-contained finite deterministic-test fixture: compressed integer mask/budget
DP, canonical key and independent bottom-up checker derived from RV22-N.
Duplicated for reproducibility isolation. This is not a native private-archive
experiment or a production proof importer.

Start cold, sample a rotating pair of query positions per probe period,
and use only sampled canonical-key digest hints to infer reuse. A fixed
16-entry packed history (192 bytes; tag uint64 and time uint32) and six
coefficients have a 256-byte accounting reserve, including a conceptual
allowance for scalar state. This does not bound actual Python object memory
or transient DP scratch. Cache keys plus serialized policies are limited to
8192 minus reserve bytes, checked per query. Matched static LRU controls
reserve the same amount. A cold query outside the probe schedule skips full
canonicalization. Retain caches across mode changes. On a probe, full-key
equality and exact replay are mandatory for cache reuse. Checked switch arms
admit only after an observed digest repeat and independent Bellman check.
Trusted and checked caches remain separate throughout. A digest collision
can affect mode/admission but cannot authorize a false cache hit.

Analytic switch activates on a recent sampled repeat and leaves active mode
after ten queries without a repeat. Learned switch uses an additive ridge
head with features bias, repeat, recent count, gap, EWMA and interaction.
Targets are future 12-query repeat fraction relative to the recent window or
earlier future queries. Training uses 36 streams of 72 queries from burst,
mixed and churn, yielding 1492 sampled feature rows and 1035 canonical keys.
This is a workload-reuse proxy, not a direct estimate of net CPU savings.
On six separate development streams, choose (period,idle,threshold) from
(4,8,.25),(8,10,.25),(8,10,.4),(12,12,.4) by summed CPU across trusted and
checked learned arms. Selected (8,10,.25). Timing calibration can select
differently across runs. Analytic fixed at (8,10,repeat).

Fresh evaluation: eight streams per cohort, 48 queries, seven cohorts,
nine arms and two randomized arm-order timing repetitions. Train seeds
2026191000+, calibration 2026192000+, evaluation 2026193000+ with
100*cohort+ordinal offsets. No canonical-key overlap between train,
calibration and any eval cohort, or among eval cohorts. Four eval workload
patterns (phase, recovery, scan, delayed) absent from training, although
phase/recovery/scan were explored in earlier studies. Delayed reuse rotates
nine projects and inserts periodic novel queries. Exact serialized canonical
keys checked; no general isomorphism claim. All 48,384 arm-query executions
(5376 per arm, 2688 distinct query positions) agree exactly on utility.
CPU process time includes solving, checking, key construction, monitoring and
replay, excludes offline training and final evaluation. Paired stream bootstrap
10000 draws over eight streams after averaging two repetitions per stream.
No adjustment for multiple comparisons or transfer across machines.

## Results

Mean CPU milliseconds per 48-query stream:

| Cohort | Cold | Trusted matched | Trusted sample | Trusted LDT | Checked matched | Checked sample | Checked LDT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burst | 35.746 | 6.676 | 7.378 | 7.045 | 11.294 | 16.628 | 15.984 |
| Mixed | 34.739 | 10.063 | 20.134 | 10.492 | 19.037 | 28.501 | 24.947 |
| Churn | 35.902 | 37.032 | 36.739 | 37.746 | 72.420 | 37.450 | 39.139 |
| Phase | 32.461 | 20.764 | 21.507 | 21.260 | 39.860 | 24.987 | 25.524 |
| Recovery | 41.322 | 21.792 | 27.634 | 22.291 | 41.229 | 31.637 | 28.873 |
| Scan | 34.134 | 25.997 | 28.005 | 26.517 | 49.958 | 30.515 | 29.504 |
| Delayed | 35.873 | 13.603 | 19.605 | 13.766 | 25.068 | 31.606 | 27.529 |

Checked sample churn constructs 12 keys, bypasses 36, admits zero entries
and reduces CPU by 48.3% vs matched checked LRU, but costs +1.548 ms
[.871,2.229] versus cold. Checked LDT constructs 33.75 keys, makes .625
admissions, and costs +3.237 [2.024,5.145] vs cold. Low threshold and positive
intercept allow activation even before a repeat, accounting for costly
false activations. LDT also reduces checked CPU vs analytic on mixed
(-3.554 [-6.325,-1.075]), recovery (-2.763 [-3.312,-2.251]) and delayed
(-4.077 [-4.599,-3.502]); loses on phase (+.537 [.333,.752]).
The learned switch is faster than cold on six of seven cohorts, but does
not dominate either the analytic switch or static checked cache.
Trusted matched LRU has lower mean CPU than trusted LDT in all cohorts.
No shuffled head or matched always-probe second-observation baseline in
this campaign, so no isolated learned-specific advantage or causal
claim separating bypass from second-observation checking. No promotion.

## Verification

Two local pytest functions passed. Independent index/used-test oracle on
24 worlds, 24 corrupted Bellman roots rejected, full-key hit safety under
forced digest collision, all arms activated, payload cap and no-storage
fallback. Mocked canonical constructor confirms exactly 12 calls and 36
bypasses on a 48-query novel-only stream.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/cold_bypass.py --out /tmp/rv22p.json

Measured source SHA256 533051a9582094a019de04c980c6f69b3c9a7ee73a774a7dd6bdffd55323e5d7.
Measured source Git blob bc7aff2a8961e1ea380dc1bc9acc812e2138face.
Test Git blob 83dd2b534c641cabc721ce2dfcc3b950c097ede8.
Receipt SHA256 92b1b7dff1e4c8a18428e73d868350eeb875a1ae9b8cc03a4c0cb746655a7fa3.
Hosted CI pending.

## Next

RV22-Q: distinguish activation from admission. Compare always-key
second-observation, sampled second-observation without mode activation,
two-confirmation analytic switching, and learned switching with hard
repeat evidence. Fit actual incremental cost labels and include shuffled
controls, fixed operating thresholds and an adversarial sparse-probe alias
family. Require lower total CPU under churn without materially sacrificing
stable reuse; if not achieved, preserve the cheap static default.
