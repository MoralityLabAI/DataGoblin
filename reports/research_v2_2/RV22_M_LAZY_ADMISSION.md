# RV22-M: lazy admission and selected continuation caching

Completed local CPU study. No model endpoints, GPU or paid compute. Parent
RV22-L: 790 passed / 15 skipped; both hosted workflows green.

## Method

Solve with the compressed integer mask DP. Independently check every Bellman
record in increasing budget order, recomputing class masses and the maximum
over all legal first tests. Pure-class stopping is justified by positive test
costs and the inherited reward ordering. Export only the selected policy
continuation(s) with normalized canonical key, exact numerator and policy.
On lookup, compare full serialized keys and replay original hypotheses.
Ordinary trusted solver entries and independently checked entries are distinct
experimental arms. This is in-process checked memory, not a portable proof
format; an authenticated external proof importer is not implemented.

Each arm uses an 8192-byte cap on retained serialized keys plus payloads,
with LRU eviction; the 128-entry digest history, Python container overhead,
temporary DP and proof-check allocations are excluded. Digests are used only
for routing hints. Total CPU per trajectory includes lookup, canonicalization,
solving, replay, proof checking, selected-state export and eviction.

Learn a seven-coefficient additive ridge head on 72 streams (2,880 query
observations) to predict occurrences of the same canonical root in the next
12 queries. Features: bias, clipped prior count, clipped recency, adjacent
repeat flag, hypothesis count, test count, budget. This approximates reuse
frequency rather than cost-adjusted utility or descendant reuse. Select a
threshold from .5,1,1.5,2,3 on a separate 18-stream timing calibration.
Threshold 1.5 selected; freeze for the subsequent nine-arm experiment.

Six-arm development: 48 streams, 1,920 queries per arm, exact utility parity.
An intermediate expanded run had inactive branches for three new controls
and was discarded. Activation assertions were added and all arms rerun on
fresh seeds 2026114500+100*cohort+ordinal. Final evaluation: 16 streams per
cohort, 40 queries per stream, 1,920 queries per arm, 17,280 total arm-query
executions. All exact integer numerator/denominator pairs agree. Distinct
generation seeds, not an exhaustive structural isomorphism audit.
Arm order shuffled per trajectory; 10,000 paired bootstrap draws over streams.
No adjustment for multiple comparisons or hardware timing transfer.

## Results

Mean CPU milliseconds per 40-query stream:

| Arm | Burst | Mixed | Churn |
|---|---:|---:|---:|
| Cold DP | 22.822 | 29.808 | 29.689 |
| Raw identity LRU | 13.944 | 21.577 | 28.260 |
| Canonical root LRU | 9.207 | 12.895 | 28.322 |
| Canonical selected states | 21.311 | 39.500 | 85.952 |
| Independently checked root | 16.790 | 23.320 | 61.630 |
| Checked root on second observation | 20.436 | 25.802 | 33.976 |
| Eager checked selected states | 29.914 | 56.920 | 134.261 |
| Lazy checked selected states | 27.543 | 36.570 | 38.182 |
| Learned checked selected states | 30.154 | 31.791 | 32.598 |

Canonical root saves 59.7% and 56.7% CPU in burst/mixed. Independently checked
root saves 26.4% and 21.8%. Paired checked-root CPU deltas versus cold
[-7.947,-4.082] ms and [-13.760,-0.657] ms (95% percentile intervals).
Churn checking costs substantial extra CPU. Learned selected-state admission
reduces eager proof admissions (burst 11.25 to 6.81, mixed 19.19 to 4.56,
churn 40 to 0). It remains slower than cold on all three cohort means.
The burst and churn paired intervals exclude a speedup; mixed is inconclusive.
Root-only check control has identical hit counts to trusted canonical root.
Eager selected descendants crowd useful roots out under the payload cap.
No learned-memory promotion.

Measured source Git blob 265407d89ccf4cd930aba891e13ae893e5831c39.
Measured receipt SHA-256 f5b269fd5e33ebdc3a2dcccb172132f25d23bce80c65dc06abdbe342e26a75cf.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/research_v2_2/lazy_admission.py --out /tmp/rv22m.json

Next: compare cheap analytic and learned root-only admission with expected
saved CPU labels (including actual miss cost and eviction opportunity cost).
Measure resident byte usage and avoid admission of low-utility terminal
subtrees; quantify prospective savings under fresh workload-family shift.
