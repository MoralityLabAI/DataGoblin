# RV22-W: finite-horizon checked memory

Local CPU-only synthetic experiment, no promotion or paid inference. Parent
RV22-V: 810 tests passed, 15 skipped; dedicated verifier workflow green.
General CI failed only B023 (resident closes over now); bind the loop variable
as a default argument, without changing the synchronous call semantics.

## Method

Freeze DGK/1, the RV22-P solver and checker, and the RV22-V history and
arena classes. All non-cold arms reserve the same 547-byte Python bytes
object (514-byte lifetime table) in addition to the packed 64-slot history.
Arena capacity is reduced accordingly. Charge the retained objects and
their buffers via sys.getsizeof under the 4096/8192 byte caps. Temporary
parsed rows, solver, per-query scratch, receipt, static code, interpreter
and allocator/RSS are outside this cap. No external certificate importer.
Full binary-key equality and original-hypothesis replay authorize hits;
digest observations are advisory only. Independently check each admission.

Draw 256 independent synthetic termination lengths from {48,72,96,144,192}
with relative probabilities {1,2,3,3,1}, seed 2026720000. Store uint16
counts S(t)=number of lengths >= t for t=0..256. Counts at t=1,49,73,97,
145,193: 256,226,188,121,31,0. No world identities in the training table.
Conditional expected periodic repeats after observing query t with gap d:
sum over j>=1 of S(t+jd)/S(t). If S(t)=0, forecast zero. For resident
entries first predicted gap is max(1,d-age), and age decay is
clip(2-age/d,0,1). This is a periodicity proxy, not a fitted recurrence
model. Conservative lower arm subtracts sqrt(log(2*257^2/.05)/(2*S(t)))
from each conditional survival probability, truncating at zero.
Conditional on each at-risk sample size, one-sided Hoeffding and a union
bound over age pairs yield a conservative simultaneous envelope for
the IID training lifetime distribution. That does not guarantee a lower
bound on actual repeat count under regime change or workload shift.
Age decay and periodic recurrence remain heuristic.

Ten arms: cold, second-observation LRU, third-observation LRU, delayed
freeze, fixed 96-query economic forecast, fixed + age decay, fixed +
age decay/reset, survival, conservative survival, and diagnostic
known_end. Only known_end is given total query count for admission;
it uses exact remaining periodic opportunities, not future root IDs
or a full clairvoyant cache optimizer. All economic arms use measured
logical solve edges for value and fixed check price 1.2 times solve
edges (a rounded proxy chosen in advance from prior checker ratios).
No elapsed CPU time enters admission. Victims sorted by predicted
saved work per stored byte; admit only if gross expected saved solve
edges exceeds check price plus predicted displaced value. Hits update
interval with max(current,previous), all histories long. Zero-edge roots
may be declined by the economic gate. Record deterministic weighted
work = 5*solve_edges+6*verify_edges; this excludes key, replay, cache
and model costs, which are included in process CPU time. No claiming
logical work equals wall time.

## Campaign and results

Six independent cohort invocations, fresh base seed 2026730000+
100*cohort+ordinal, three streams per family, lengths as below,
two capacities and ten arms with randomized arm/cap order per stream.
No timing repetitions. 18 streams, 2220 query positions, 44,400 arm-query
executions, exact numerator/denominator agreement throughout.
A post-run audit reconstructed every stream and established 580 distinct
full binary keys with zero intersections between streams and cohorts.
The lifetime training data are independent length samples, not roots.
Short12 and long12 include termination outside training support
(24/36 and 216). All evaluation roots are spent. Small, heterogeneous
three-stream cohorts, fixed lengths rather than IID lifetime draws.
Bootstrap over three streams only, exploratory timing, no multiplicity
adjustment. Timing includes arena parse/rewrite, monitor, forecast,
key, solve, verification and replay, excludes offline inventory/model
construction. Cross-cohort CPU clock conditions can differ.

4 KiB mean CPU ms per stream:

| Cohort (lengths) | Cold | Second | Third | Freeze | Fixed | Fixed age | Fixed reset | Survival | Lower | Known end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short12 (24/36/48) | 34.82 | 57.03 | 45.93 | 51.46 | 34.79 | 39.00 | 35.94 | 37.08 | 38.34 | 32.82 |
| Long12 (120/168/216) | 161.33 | 311.94 | 298.80 | 81.47 | 48.06 | 48.24 | 47.97 | 48.29 | 49.63 | 52.92 |
| Cycle40 (96/128/160) | 120.82 | 221.76 | 183.54 | 142.69 | 130.26 | 138.65 | 127.32 | 129.33 | 129.47 | 126.51 |
| Cycle56 (120/144/192) | 137.54 | 235.29 | 178.40 | 158.09 | 149.99 | 150.46 | 155.53 | 150.69 | 146.60 | 143.03 |
| Phase (96/144/192) | 138.07 | 254.91 | 240.84 | 137.39 | 97.56 | 81.37 | 82.24 | 85.95 | 81.33 | 83.57 |
| Mixed scan (72/108/156) | 81.56 | 100.26 | 87.53 | 78.24 | 61.59 | 79.03 | 78.56 | 84.40 | 69.13 | 68.45 |

At 4 KiB, long12 second/third have no hits and 156/144 checks on
average; survival gets 125 hits and 12.33 checks, weighted edges .255
of cold. Fixed and survival have identical logical counts there. At 8 KiB
all second-observation policies fit the long12 working set and achieve
144 hits and 12 checks; third has 132 hits. At 4 KiB phase, fixed age
and reset get 78 hits, 23.33 checks vs fixed 53 hits and 16 checks;
weighted work .561 vs .666 of cold. Survival and lower also get 78 hits
with 22.67/22 checks. Reset adds no logical advantage over age alone.
On cycle56 survival and lower check zero entries, with weighted work
equal to cold; overhead means no CPU win. Cycle40 lower similarly
declines all admissions, while survival checks 11.67 times and gets
16 hits at 4 KiB. Mixed scan favors fixed over survival. Short12 at
4 KiB fixed/survival/lower identical logical work 1.099 of cold despite
timing differences; known_end gets .922, four checks and 7.33 hits.
No general survival-controller promotion. The prior support ends at 192,
so later cache misses cannot justify new survival admissions without
a tail model.

## Identifiability boundary

With space for one certificate and no eviction opportunity cost,
let verification cost v>0, saving per later reuse s-h>0. At an eligible
admission decision, consider two continuations sharing the same
observable prefix: immediate termination, or N further identical
queries with N(s-h)>v. Immediate admission strictly loses v in the
first continuation; in the second it is cheaper than never admitting
and cheaper than delayed admission by at least one saved solve under
constant costs. An online policy seeing only the shared prefix cannot
make the offline-optimal choice in both cases. Randomization cannot
guarantee offline-optimality on both. A survival prior is a workload
assumption, not a universally sound economic guarantee. The exact
utility guarantee survives all admission decisions.

## Verification and reproduction

Three local pytest functions passed: survival sums against direct
finite lifetime enumeration; conservative forecast bounds; exact
cross-arm replay, check/admission and weighted edge accounting;
shared-prefix check traces for online arms with different stream
lengths; forced digest collisions and zero-storage fallback. No new
independent solver was added; relies on previously audited DP/checker.
Parent lint fix included. Hosted RV22-W CI must be checked separately.

Measured source SHA256 5e2114d0c4d3457115afdcfd2dd5b45d012729fc586ce101de3b749d96a7083c.
Source blob 49ee2419bb4f421f4e14a7bbd6e28209e304ef11.
Test blob 8601d7666dab34723754ebb753dee955b7c8e4fd.
Combined local receipt SHA256 975fec55366608b22c94b1a404e190cca3ee6ba241896c3f3593dae221fca9f7.

Reproduce each cohort (short12,long12,cycle40,cycle56,phase,mixed_scan):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/finite_horizon.py --cohort short12 --out /tmp/rv22w_short12.json

## Next

RV22-X: unknown-horizon rent-or-buy admission with an explicit
per-key payback counter and a bounded overflow-aware replacement rule.
Derive competitive bounds only under a stated no-eviction/constant-cost
model, then test against a finite offline cache oracle on small streams.
Measure actual CPU separately from deterministic work. Include abrupt
termination and adversarial working-set changes. Stop if simple analytic
admission remains preferable; do not train a new learned controller.
