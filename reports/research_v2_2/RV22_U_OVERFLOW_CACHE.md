# RV22-U: scan-resistant checked memory under overflow

CPU-only exploratory synthetic campaign. Parent RV22-T 76444c38 passed both
hosted workflows. Freeze the DGK/1 binary codec (blob
4ac15e8c69fed2cb965e6152005e84ea15803b98); full key equality and exact replay
authorize hits. Every admitted entry is independently Bellman-checked.
No learned component, paid inference, GPU use or promotion.

## Design and accounting

A shared 32-query sliding digest history supplies prior frequency; the
digest is advisory only. Six policies: cold DP; eager checked LRU;
second-observation checked LRU; second-observation freeze (reject rather
than evict); strict frequency guard (sort victims by current window count,
and admit only if prior candidate count exceeds their total); economic
guard (sort by estimated remaining savings per serialized byte and admit
only if candidate's prior count times measured miss saving exceeds predicted
checking cost plus foregone victim savings). Ties in victim ordering retain
LRU order. No check is performed for rejected admissions. The strict count
comparison intentionally makes equal-frequency replacement difficult.
All non-cold arms use identical 1024-byte logical history reserve and 24
bytes per entry for metadata plus actual serialized key and policy size.
The reserve is an accounting allowance, not a packed implementation or
resident RAM cap. Shallow Python size estimate charges retained cache,
history and counts separately, may double-count shared objects, and excludes
temporary solver allocations/allocator overhead. Instrumentation for this
estimate is outside the timed region. Cold control has no serialized
reserve but allocates empty containers in the fixture.

Economic calibration: 32 separate 16-hypothesis worlds, median
check/cold-solve ratio 1.1498778814, median decode/replay .022403 ms.
Future gain uses prior observed count as a repeat-count proxy; no horizon
extrapolation and no learned fit. Measured miss savings and calibrated costs
are noisy local estimates. Full canonical keys disjoint between these 32
calibration roots and all 300 distinct evaluation roots, independently
audited. Evaluation streams disjoint by full binary key.

15 streams (three per family), 72 queries each, three accounting capacities
2/4/8 KiB, six arms, single randomized arm/capacity timing order:
19,440 arm-query executions. All agree exactly on integer numerator and
denominator. Fresh seeds 2026600000+100*family+ordinal. All five cohorts
use sparse eight-test, 16-hypothesis worlds. Cycle4/12/24 are exact cyclic
working sets. Scan interleaves four recurring roots (one every third query)
with fresh roots. Phase changes to four disjoint roots halfway through.
Per-stream repeated queries use invariant variants (row and test permutation,
uniform mass scaling, duplicate split). CPU process time charges key,
history, policy, solve, verification, eviction and replay; excludes offline
calibration and retained-size measurement. Bootstrap 10,000 paired draws
over only three independent streams; percentiles exploratory, no strong
inference or multiplicity adjustment.

## Results

Mean CPU ms per 72-query stream:

| Family | Cap KiB | Cold | Eager | Second | Freeze | Frequency | Economic |
|---|---:|---:|---:|---:|---:|---:|---:|
| cycle4 | 2 | 90.36 | 13.45 | 18.49 | 17.77 | 17.70 | 23.68 |
| cycle4 | 4 | 91.63 | 13.21 | 17.37 | 18.98 | 17.53 | 22.66 |
| cycle4 | 8 | 88.02 | 12.77 | 17.92 | 18.24 | 18.72 | 22.88 |
| cycle12 | 2 | 95.28 | 186.66 | 168.86 | 82.28 | 79.71 | 86.96 |
| cycle12 | 4 | 93.54 | 34.14 | 49.38 | 49.46 | 49.38 | 64.23 |
| cycle12 | 8 | 95.70 | 34.54 | 50.23 | 50.59 | 47.54 | 63.81 |
| cycle24 | 2 | 94.69 | 181.97 | 153.32 | 95.53 | 94.03 | 98.16 |
| cycle24 | 4 | 90.16 | 185.77 | 157.24 | 92.86 | 96.96 | 94.05 |
| cycle24 | 8 | 92.72 | 63.39 | 95.03 | 93.06 | 93.35 | 97.41 |
| scan | 2 | 86.66 | 183.78 | 106.53 | 78.34 | 78.23 | 81.75 |
| scan | 4 | 89.69 | 130.71 | 81.05 | 75.18 | 74.84 | 82.05 |
| scan | 8 | 89.71 | 127.71 | 76.26 | 76.44 | 81.69 | 79.96 |
| phase | 2 | 90.50 | 24.45 | 33.05 | 62.79 | 49.78 | 56.78 |
| phase | 4 | 89.44 | 23.30 | 32.67 | 32.90 | 33.53 | 43.18 |
| phase | 8 | 89.99 | 23.49 | 32.76 | 33.24 | 33.47 | 42.92 |

Cycle12/2 KiB: second 60 checks, zero hits; freeze and frequency 4 checks,
16 hits, no evictions. Frequency minus second -89.152 ms
[-96.086,-85.346]; minus cold -15.575 [-21.328,-11.049].
Cycle24/4 KiB: second 48 checks, zero hits; freeze/frequency 12.33
checks and hits each. With only one remaining cycle after admission
the saved solve approximately cancels the expensive verification.
Cycle24/8 KiB: eager 48 hits, 24 checks; second 24 hits, 24 checks.
Cycle24 economic zero checks and hits at all capacities. Since reuse
distance 24 and history 32, prior count is at most one. Candidate gross
gain <= current miss cost, but predicted check cost exceeds miss cost.
The rule therefore never admits, regardless of subsequent repetitions.

Scan/2 KiB: frequency 13.33 hits, 3.33 checks, vs second 5.33 hits and
14.67 checks; frequency minus second -28.297 ms [-40.171,-5.390],
minus cold -8.427 [-10.211,-6.231]. Phase/2 KiB: freeze has 28 first-half
hits and zero second-half hits. Frequency recovers total 43.67 hits
but still trails delayed LRU's 56; frequency minus second +16.725
[15.745,17.665]. This is a real adaptation tradeoff.

Illustrative retained-object estimates: cycle12/2 KiB frequency 6230 B,
second 6439 B; cycle24/8 KiB eager 17945 B; scan/8 KiB eager 22791 B.
Thus logical capacities are materially smaller than the Python footprint.
No RSS or RAM-cap claim.

## Analytic explanation

For R complete equal-size cycles of d roots with capacity for C<d,
assuming each root is observed in the history on its second visit,
eager LRU has zero hits, Rd checks; delayed LRU has zero hits,
(R-1)d checks. Freeze checks C roots once during cycle two and then
has (R-2)C hits, with no evictions. Under equal constant cold-solve cost
s, hit cost h and check cost v, its cost minus cold is
C v - (R-2) C (s-h), excluding key/history overhead. Savings become
positive only when (R-2)(s-h)>v. This proves the capacity-cliff mechanism
for the equal-size cyclic model, not for heterogeneous workload streams.
Strict frequency acts like freeze on the tested homogeneous cycles but
can admit new roots after old evidence decays. Measured results do not
support a universal gate.

## Verification and next

Two local pytest functions passed: every arm exact utility parity,
check/admission and byte accounting, overflow regression, freeze and
frequency phase response, forced digest collisions, and zero-payload
fallback. No new external proof parser or independent solver was added;
reuse the previously audited RV22-P/T fixture and checker. Hosted CI
for this commit remains unverified. Measured source SHA256
0000090caaab80a957228df17d7273ed2d1a03b5328a245357d8c00543b6ef9a.
Source blob 171f8967c08e55d37fb0c12218fbec5291d0b21f;
test blob ffc9fc018fce7ceff469b303c4d934b9affe124e.
Receipt SHA256 697fd56ce2632ea08c62c7a0c0da297afd8bfc0b9fd99fcec1ca741dc96a187f.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/overflow_cache.py --out /tmp/rv22u.json

RV22-V: derive a reuse-distance estimator with bounded storage and separate
admission/eviction evidence. Include a regime-change reset and measure long
cycles beyond the 32-query history, mixed entry sizes, and matched actual
retained-object budgets. Compare deterministic policies before fitting
anything. Preserve checked trust and exact full-key replay.
