# RV22-X: unknown-horizon rent-or-buy and finite offline cache oracle

CPU-only local exploratory experiment. RV22-W cd7065aa passed both hosted
workflows. Frozen DGK/1 codec and RV22-P exact solver/checker, RV22-V arena.
No training, paid inference, GPU use, or promotion.

## Model and theorem

For a single key with constant rent r>0, buy cost b>0, zero logical hit
cost, unlimited exact history, no eviction, no prefetch, and a mandatory
rent on the purchase request, let the online policy buy once accumulated
*prior* rent is at least b. Its purchase request is j=ceil(b/r)+1.
For N requests, OPT=min(Nr,r+b): if buying at all, buying on the first
request minimizes cost. For N<j, Nr<=ceil(b/r)r<=b+r, so the online policy
is optimal. For N>=j, OPT=r+b and the online cost jr+b<=2(r+b).
Hence the policy is 2-competitive. This extends by summation to independent
keys that all fit and have accurate persistent histories. It supplies no
bound for finite-capacity eviction, tag collisions, saturation, changing
costs or positive lookup/monitor/hit overhead. Past rent is sunk work;
the policy is a threshold heuristic for uncertain future benefit.

Offline oracle: request t specifies exact identity, fixed serialized record
size, rent and buy costs. Memoize (t,resident bitmask). Hit costs zero;
on miss choose bypass (pay rent) or buy (rent+check), after removing any
subset of resident entries to fit. Free eviction on a hit/bypass can always
be postponed until the next purchase without increasing cost. Exhaustive
subset enumeration gives the exact minimum under this finite model.
At most 12 keys and 96 requests. No claim of a scalable offline algorithm
or a clairvoyant optimal CPU scheduler.

## Implementation

64 fixed 16-byte history records store digest hint, uint16 timestamp/count,
and saturated uint32 accumulated rental work. On a miss add 5*solver.edges
unless buying, in which case reset debt. Payback requires positive rental
work and prior debt >= 6*solver.edges. Payback_freeze forbids displacement;
payback_guard additionally charges decaying victim buy-cost hurdles
(1.2*recorded rental * clip(2-age/interval,0,1)), orders by hurdle per
record byte, and requires candidate debt >= buy + sum(hurdles).
This is heuristic hysteresis rather than a proved economic value.
Controls cold/eager/second/third/freeze use the same exact-key arena and
ledger (cold has no cache). Full binary key equality plus replay authorizes
all hits. Every admitted entry is independently Bellman-checked. Digest
collisions affect admission only. The logical work model prices solving
at five units per solver edge, checking at six per verifier edge, and
omits hit/lookup overhead. Admission decisions use no CPU timing or
future termination length. Actual process CPU separately charges all
online work including exact key, lookup, replay and arena rewriting.
All cache arms use the same fixed retained objects/buffers cap; no
survival-table reserve in RV22-X, so capacity is not identical to W.
Parsed rows, solver scratch, decision trace/receipt, interpreter and
allocator/RSS are excluded. All inputs are valid synthetic fixture worlds.

## Offline audit

32 streams, eight each cycle/burst/scan/shift; 1011 request positions,
seed 2026750000; eight variable-size roots each; cap 2048 or 2304 bytes
including ledger and arena. Derive logical rent/buy and record size from
exact solver and policy. All online aggregate costs are >= exact offline
optimum; this audit checks logical cost, while the main campaign checks
cross-arm utility equality explicitly. Offline DP visited 101492
(t,mask) states total. Ratios below are sums of costs divided by sums
of oracle costs within each family, not mean per-stream ratios.

| Arm | Cycle | Burst | Scan | Shift |
|---|---:|---:|---:|---:|
| Cold | 1.529 | 1.781 | 1.486 | 1.793 |
| Eager | 3.364 | 1.029 | 3.269 | 1.783 |
| Second | 2.905 | 1.497 | 2.756 | 1.965 |
| Third | 2.446 | 1.646 | 2.115 | 2.148 |
| Freeze | 1.384 | 1.705 | 1.236 | 1.623 |
| Payback | 1.786 | 1.646 | 1.794 | 1.892 |
| Payback freeze | 1.565 | 1.745 | 1.413 | 1.835 |
| Payback guard | 1.673 | 1.646 | 1.634 | 1.891 |

Maximum per-stream ratio: payback 1.921; guard 1.909; payback freeze
2.160; second 3.282; eager 3.800. Observed ratios do not prove a
finite-capacity competitive bound.

## Fresh campaign

18 streams, three per family, 1710 query positions, eight arms and two
caps (4096/8192), 27360 arm-query executions with exact integer
numerator/denominator agreement. Each family invoked separately, one
randomized arm/cap order per stream. Seed 2026740000+100*family+ordinal.
Post-run full-key inventory audit found 549 distinct keys and zero
cross-stream/cross-cohort intersections within the main campaign.
The offline audit uses a separate seed range; overlap with that audit
was not explicitly checked. Terminating cohort begins with repeated
12-root cycles and switches to entirely novel queries halfway through.
Small exploratory cohorts, no timing repetitions or confidence intervals.

Mean CPU ms at 4 KiB:

| Family (lengths) | Cold | Eager | Second | Third | Freeze | Payback | Payback freeze | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short12 (24/36/48) | 31.32 | 23.48 | 33.50 | 39.93 | 34.16 | 37.84 | 37.67 | 37.84 |
| Long12 (96/120/144) | 116.95 | 30.32 | 42.71 | 53.99 | 45.39 | 52.52 | 52.71 | 53.80 |
| Cycle40 (96/120/144) | 106.59 | 238.58 | 188.54 | 156.80 | 112.27 | 149.37 | 124.72 | 132.71 |
| Phase (96/120/144) | 116.76 | 57.44 | 78.13 | 102.03 | 101.68 | 101.72 | 112.66 | 110.84 |
| Scan (72/96/120) | 74.55 | 105.65 | 60.38 | 73.42 | 57.40 | 66.64 | 66.82 | 66.58 |
| Terminating (54/78/102) | 70.16 | 88.65 | 62.05 | 71.27 | 63.50 | 73.11 | 75.80 | 67.38 |

At 4 KiB the 12-root sets fit. Eager/second/third weighted logical work
on long12 is .219/.319/.419 of cold; all three payback arms equal
third. Cycle40 delayed LRU checks 80 entries with zero hits;
freeze checks 14.33 and hits 19, weighted work .988 of cold.
Payback checks 32 with zero hits; guard checks 19.67 with 3.33 hits.
Their weighted work 1.317/1.172 of cold. At 8 KiB cycle40 freeze
has 35 hits and 32.67 checks, yet costs 1.042 of cold; all payback
variants also lose. Scan freeze and second achieve 40 hits and
12 checks, logical work .740; third/payback 28 hits and 12 checks,
.865. Terminating second is slightly above cold in logical work
(1.008); payback is 1.060, despite some favorable noisy CPU times.
The policy delays useful admissions without reliably protecting from
overflow. Freeze is the stronger simple overflow reference in this
fixture, eager for fitting/recurrent working sets, and cold for no
profitable reuse. Stop the learned-memory escalation; the bounded
analytic result is enough to motivate workload-dependent integration
rather than yet another gate. No universal default or promotion.

## Verification and reproduction

Two local pytest functions pass. Independent forward subset DP agrees
on 120 variable-cost, variable-size synthetic traces. 7830 arithmetic
checks of the no-eviction bound and 1044 single-key oracle comparisons.
Shared-prefix decisions, third-visit purchase under fixture cost,
exact replay, forced digest collision, check/admission and memory
accounting, and zero-entry budget fallback verified.
No new independent research-value solver. Latest hosted CI unverified.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/rent_or_buy.py --offline --out /tmp/rv22x_offline.json
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/rent_or_buy.py --cohort cycle40 --out /tmp/rv22x_cycle40.json

Source SHA256 414bc67d926556f4c9a78b6940dcf878c6dd2cdaa3acebd842e7d504ef02f2a5.
Source Git blob ceca3a49c15207e81e0bab347286ca865256cb2b.
Test Git blob f4f5535fff1f6ae15079fc6596eb7634aade032c.
Combined receipt SHA256 7420046bb6334ca81f2f79b0d57e73ab4a86dba8cf51234b70214ae23ee5075b.

## Next

RV22-Y: close the synthetic memory branch with a frozen capability and
failure ledger. Integrate only explicitly selected exact solver/cache
components behind a strict input/trust contract; return to native Distill
decision-value work and the pending actor/Baby Goblin evidence gates.
No production import, trained memory controller, or checkpoint
promotion inferred from these synthetic experiments.
