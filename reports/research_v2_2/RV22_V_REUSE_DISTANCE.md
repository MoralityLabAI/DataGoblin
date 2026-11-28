# RV22-V: recurrence-aware checked memory with packed allocation

Completed local CPU-only synthetic experiment. Parent RV22-U 551574ef
passed both hosted workflows. Frozen RV22-T binary key codec (blob
4ac15e8c69fed2cb965e6152005e84ea15803b98), RV22-P exact DP and checker
(blob bc7aff2a8961e1ea380dc1bc9acc812e2138face). No paid inference, GPU,
learned model or promotion.

## Method

The 64-slot history uses 16-byte packed records (uint64 digest, uint16
last-observation index, uint16 previous interval, uint8 saturating count,
padding) and an 8-byte regime header. The short controls maintain an
event ring and query a 32-query window; long controls maintain the
64 most recently observed distinct tags and track their previous
interval. This measures temporal recurrence interval, not conventional
distinct-key stack distance. Collisions or distinct-tag overflow can
change admission but cannot authorize reuse. All hits require the full
DGK/1 binary key and original-hypothesis replay.

A preallocated bytearray holds variable-length key and policy bytes plus
16-byte record metadata (lengths, last access, interval, measured miss
saving), with a 4-byte arena length header. Both state objects use
__slots__; the budget is the sum of sys.getsizeof of the two instances
and their two bytearrays. All six cache arms have identical fixed
allocations of 4096 or 8192 bytes on local CPython 3.13.5. The accounting
is stronger than a serialized allowance but limited to these retained
objects; parsed records, transient bytes, solver scratch, evaluator
receipt, other Python locals, static code and process allocator/RSS
are excluded. Parsing and rewriting the arena is timed. A production
arena may avoid copying unchanged records; this prototype does not.

Cold DP, short-window second-observation LRU, short strict frequency
guard, long second-observation LRU, long freeze, distance economic guard,
and distance + age decay/reset. All cache entries independently
Bellman-checked before admission. Long-distance candidate estimated
interval max(current interval, previous interval, 1), projected repeat
count 96/interval, and saving = max(0, measured cold solve minus estimated
decode/replay). Resident value = 96/resident interval times its recorded
saving. Economic policy ranks eviction candidates by value per record
byte and accepts only when candidate projected gross saving exceeds
estimated check cost plus displaced resident value. No horizon truncation
at stream end, which is not known to the actor. Age-aware variant scales
resident value by clip(2-age/interval,0,1). After four consecutive repeat
observations arm a reset detector; eight consecutive novel observations
trigger removal of residents older than twice their intervals. Reset
does not erase the digest table. No learned fit.

Cost calibration on 32 independent alternating 12/16-hypothesis worlds:
median check/solve ratio 1.1355422, median decode/replay .021802 ms.
This does not include all hit overhead (arena parse, lookup and rewrite).
Per-query solve timing and these calibrated costs are noisy economic
proxies. CPU is process time, includes key generation, arena and history,
solver, checker and replay, excludes calibration, offline inventory and
receipt aggregation. Timing randomizes arm/cap order once per stream.

## Campaign

15 fresh streams, three per family, 144 queries, two capacities, seven
arms (30,240 arm-query executions). Seed 2026710000+100*family+ordinal;
separate calibration seed 2026700000. All 576 distinct evaluation
canonical keys and 32 calibration keys are mutually disjoint across
streams and calibration (not a general structural isomorphism audit).
Cyclic families have recurrence 12, 40 or 56; phase changes between
disjoint 12-root sets at query 72; mixed scan repeats adjacent variants
of 12 roots interspersed with one novel query in every three. Mixed
12/16-hypothesis, six/eight-test worlds give variable entry sizes and
solve costs. Every arm agrees exactly on each numerator/denominator.
No tuning after smoke. Paired stream bootstrap 10,000 resamples over
three independent roots per family; exploratory intervals, no multiple
comparison correction or hardware generalization. No eager checked
LRU arm in this comparison (see RV22-U).

## Results

Mean CPU ms per 144-query stream:

| Cohort | Cap KiB | Cold | Short second | Short frequency | Long second | Long freeze | Distance | Distance + age/reset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cycle12 | 4 | 148.53 | 47.49 | 49.44 | 43.77 | 45.32 | 44.30 | 44.79 |
| cycle12 | 8 | 142.38 | 45.67 | 48.57 | 50.76 | 44.53 | 45.81 | 44.78 |
| cycle40 | 4 | 131.97 | 146.02 | 137.29 | 240.72 | 131.83 | 129.34 | 130.96 |
| cycle40 | 8 | 130.86 | 137.92 | 141.41 | 241.35 | 121.93 | 123.87 | 127.54 |
| cycle56 | 4 | 131.65 | 139.28 | 135.90 | 218.80 | 141.00 | 141.85 | 147.46 |
| cycle56 | 8 | 131.23 | 138.91 | 137.67 | 221.05 | 140.74 | 149.85 | 151.28 |
| phase | 4 | 138.19 | 110.42 | 92.93 | 80.50 | 111.41 | 95.53 | 80.52 |
| phase | 8 | 137.24 | 79.13 | 81.92 | 81.55 | 80.76 | 77.20 | 83.88 |
| mixed_scan | 4 | 115.38 | 75.42 | 76.05 | 76.79 | 75.18 | 75.09 | 80.92 |
| mixed_scan | 8 | 113.97 | 76.02 | 76.52 | 77.67 | 78.66 | 79.53 | 75.63 |

All long-history arms observe all 104 and 88 available repeats in
cycle40 and cycle56 respectively; short controls observe none. At
cycle40/4 KiB, long second checks 104 times with zero hits. Freeze
checks 12.67 times and hits 25.33; distance checks 16.33 and hits 23.67.
Distance vs cold -2.631 ms [-6.549,2.372], inconclusive. At 8 KiB,
freeze checks 32, hits 56, and beats cold by -8.924
[-11.691,-5.377]; distance -6.985 [-11.206,.567]. The simple freeze
control remains strong.

Cycle56/8 KiB: long second checks 88 times and has zero hits, freeze
checks 32 and hits 31.67, distance checks 36 and hits 26.67.
Distance vs cold +18.620 [7.512,27.964] ms; freeze +9.513
[4.760,13.320]. A fixed 96-query horizon does not match realized
remaining reuse in these finite streams. Recovering long-interval
evidence can be counterproductive when certification is expensive.

Phase/4 KiB: distance 70 hits total, 22 second-half; age-aware 96 total,
48 second-half, same as long LRU. Age-aware minus distance -15.012
[-17.577,-11.738] ms; versus long second +.013 [-4.957,3.786].
The regime reset fires once per phase stream but removes zero records
at the trigger because they are not yet older than twice their interval.
Thus the benefit is associated with continuous age-decay and altered
admissions, not demonstrated causal benefit of explicit reset.
A separate targeted regression confirms actual reset eviction on
a short-period root followed by eight novel observations.
All cache arms obtain the same 72 hits/12 checks in mixed_scan and
120 hits/12 checks in cycle12; timing differences on identical logical
patterns should not be credited to better prediction.

## Verification

Three local pytest tests passed, including 24 independent index/used-test
oracle comparisons and replay/check parity, long-history boundary,
arena round-trip and compaction, every arm activation and exact parity,
forced digest collision, explicit reset eviction, zero-payload fallback,
and actual fixed retained-object accounting. Valid synthetic fixture
inputs only. No import of untrusted arena records or certificate proof
format. Timing-dependent decisions may vary by run.
Measured source SHA256 a28f51f80a32d27356dce5e44dbdd14e79837117dd78065fc4c1002b576229d8.
Git blob 6c5daea885e2ffba98e7b73e322dc9f4a40e3e2d.
Test blob c134f68dc3abd0195f231cbc3886a90cbaeec7c9.
Receipt SHA256 7dfc46fbf625fa69910ad7ee3ca09c2f0efb9569eaef3cd9860c5b58b30f4e54.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/reuse_distance.py --out /tmp/rv22v.json

## Next

RV22-W: finite-horizon admission and stopping with unknown remaining
lifetime. Compare second-observation, third-observation, payback-aware
and conservative lower-confidence reuse forecasts under varied stream
lengths and terminating workloads. Use fixed arena and same trust mode.
Add age-only ablation, and separate timing-independent logical work
prediction from CPU timing. No promotion until benefits beat cold and
simple freeze/second-observation on fresh families.
