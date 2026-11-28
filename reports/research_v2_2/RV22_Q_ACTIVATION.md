# RV22-Q: factorized cache activation and cost-supervised switching

Status: exploratory local synthetic CPU experiment. No model promotion, no
GPU usage, no external inference or paid compute. Parent RV22-P: both hosted
workflows passed.

## Design

Separate admission (only verify a missed key after an observed digest repeat)
from activation (whether to construct keys between scheduled probes).
Always-key delayed admission, sampled delayed admission, one- and two-evidence
activation, cost-supervised activation, shuffled-target control, and jittered
one-evidence activation share the same checked admission condition and LRU
replacement. Full canonical bytes authorize a hit; replay verifies its integer
value. All non-cold arms have the same 8192-byte accounting cap less a 384-byte
reserve. This is logical serialized accounting, not process RSS.

Train on 24 streams of 64 queries, burst/mixed/churn only. For each eligible
sampled-repeat decision, clone the pre-query cache and monitor. Run two
randomized-order timing repetitions of two counterfactual 12-query policies:
force full key construction, or continue sparse probing. Both start with the
same state, and retain the same admission rule. The regression target is
sampled CPU time minus full-key CPU time. This is an intervention-window
proxy; deployment has a ten-query idle rule rather than a forced window.
Nine features: six RV22-P monitor features and hypothesis count, test count,
budget. Since all training rows satisfy the observed-repeat gate, the
intercept and repeat feature (and rate and repeat-times-rate) are collinear.
Fit ridge with penalty .5 and an equal-feature shuffled-label control.
72 eligible examples, 62 positive savings, mean 3.977766 ms. Train has 637
unique canonical keys. Calibrate threshold independently for learned and
shuffled arms on nine streams, grid 0/.25/.5/1 ms. Selected 0 and .5 ms.
Calibration has 190 unique keys. No phase/recovery/scan/delayed/alias training.

The adversarial alias workload places novel roots at every deterministic
probe position and the same recurring root at the other 75% of positions.
The jittered control independently changes the two-probe offset by block;
its fixed seeds are independent of root generation. This is a constructed
schedule-specific counterexample, not a prevalence estimate.

An initial six-stream attempt timed out before completing/writing its
receipt. Discarded those timings from the main result; reran with a fresh
evaluation seed range and four streams per cohort, 48 queries, two timing
repetitions, ten arms, eight cohorts (30,720 arm-query executions). The
rerun refits timing-dependent targets and thresholds on the original
training/calibration streams. Thus exploratory, not untouched confirmation.
Exact canonical key intersections between train, calibration and all eval
cohorts are empty. Eval cohort key counts 24/44/182/104/100/124/60/52.
Every arm/query returns the same integer utility numerator and denominator
as the other arms. Timing randomizes arm order each repetition, includes
key/monitor/solver/check/replay work, excludes offline training, clone
construction, evaluator and memory measurement after the timed region.
Bootstrap 10000 paired resamples over four streams, averaging repetitions
first. Very small n and multiple comparisons; all timing evidence provisional.

## Results

Mean CPU milliseconds per 48-query stream:

| Cohort | Cold | Trusted LRU | Checked LRU | Always second | Sample second | Switch1 | Switch2 | Cost | Sham | Jitter1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Burst | 46.581 | 6.988 | 11.921 | 16.979 | 43.750 | 17.363 | 44.323 | 16.798 | 17.077 | 17.979 |
| Mixed | 34.806 | 10.786 | 19.261 | 23.023 | 36.524 | 29.108 | 37.937 | 28.588 | 29.775 | 29.509 |
| Churn | 38.453 | 38.503 | 75.141 | 40.954 | 38.196 | 38.687 | 37.781 | 37.556 | 37.707 | 37.313 |
| Phase | 35.306 | 22.284 | 42.466 | 27.393 | 36.909 | 26.469 | 36.818 | 26.233 | 26.714 | 27.536 |
| Recovery | 41.524 | 22.549 | 41.922 | 29.779 | 45.361 | 32.379 | 44.962 | 33.191 | 32.695 | 32.284 |
| Scan | 34.031 | 25.812 | 49.760 | 30.018 | 35.357 | 30.833 | 36.065 | 30.176 | 31.162 | 36.405 |
| Delayed | 37.539 | 14.176 | 26.489 | 29.295 | 34.749 | 33.495 | 34.575 | 33.499 | 33.062 | 39.114 |
| Alias | 54.716 | 13.345 | 23.288 | 15.585 | 55.881 | 55.608 | 55.362 | 56.202 | 55.364 | 23.776 |

In churn, all sampled switches build 12 keys, perform zero checks and
obtain zero hits. Cost minus cold is -.897 ms [-5.379,2.770]; no established
speed win. Always-key delayed admission already removes nearly all
unnecessary churn checking (1 check per stream). It is often faster
than sampled switching on mixed, recovery, scan and delayed workloads.
The two-evidence rule never activates in any tested cohort: the sparse
history rarely has two prior observations of a key within 12 queries.
It offers no evidence of useful adaptive behavior.

Cost and switch1 have identical aggregate logical counts in seven of
eight cohorts; their phase counts differ (38 versus 40 keys, 3 versus
3.25 checks, equal 18.5 hits). Cost/sham likewise have identical logical
counts except phase. Apparent timing gains on otherwise matched execution
patterns cannot establish learned value. No promotion.

On alias, all deterministic sampled switches miss every recurring root,
including the learned head with its required repeat gate. Jitter1 obtains
28.25 hits per stream, 40.25 keys, one check; CPU gain versus switch1
31.833 ms (paired 95% interval [25.095,38.570]). Always-key second gets
34 hits with one check and is faster still. Jitter hurts scan and delayed
reuse: paired cost versus switch1 +5.572 [4.273,6.872] and
+5.620 [2.013,8.557] ms. Randomized/jittered probing mitigates this
specific fixed schedule alias, not an adaptive adversary guarantee.

## Verification

Two local pytest regression functions pass. All arms exercised on a repeated
root, checked admissions equal verification count, payload bound and
zero-payload fallback tested. Alias workload demonstrates 12 genuine key
calls and zero activation despite very high predicted value; jittered
probing yields hits and parity. Counterfactual copies preserve starting
cache/history and agree on exact values. Reuses source-pinned RV22-P DP,
canonicalization, checker and monitor without changes.

Reproduction:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/activation.py --out /tmp/rv22q.json

Measured source SHA256 0d9c3ac6e1998632aae436f68880cb0cff65f707c0de0afa89b862f2859f23b5.
Measured source Git blob 825b7eab22ef4a441949dc0a5066bf2b732e07ae.
Local test Git blob fe1aad322b22998e26858829bcb4bfa094039829.
Receipt SHA256 624a1a3a65bdb1a5e9111c66bcb6ae83f5325b8d190ab1e5169bbed5aa13accc.

## Next

RV22-R: stop iterating on marginal learned timing gates. Analyze sampled
observation coverage and probe alias in finite streams. Test a cheap
coarse, order/scale-invariant structural fingerprint on every query,
using full canonical equality only for candidate reuse; compare against
always-key delayed admission and jitter. Charge fingerprint overhead and
full-key false-positive fallbacks. Include forced fingerprint collisions
and hidden recurrence. Preserve separate checked/trusted trust modes.
