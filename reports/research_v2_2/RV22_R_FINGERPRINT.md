# RV22-R: invariant structural prefilters for exact research memory

Completed local synthetic CPU experiment; no external model calls, GPU or
paid compute. Parent RV22-Q (6b1ff669) passed both hosted workflows (798 tests
passed, 15 skipped; clean lint).

## Construction

Two cheap fingerprints of a finite deterministic research root:

- mass: budget, terminal rewards, and GCD-normalized A/B prior masses;
- marginal: additionally the cheapest representation of the lexicographically
  first affordable nonconstant test's cost and its two class-conditioned
  positive-bit masses, normalized jointly with the A/B masses. Test ID is
  intentionally omitted from the fingerprint tuple.

Canonical equality implies equal fingerprints: a canonical root retains all
affordable nonconstant tests in ID order and the joint class/bit distribution
up to common scale. These fingerprints are projections of that distribution.
Uniform scaling, row/test permutations, duplicate-row splitting, insertion of
constant or unaffordable tests preserve them. Neither projection is injective.
A 16-entry packed ring (192 bytes) records recent numeric fingerprint tags.
Because it holds at least the entire 12-query window and records every query,
any exact repeat in that window triggers a candidate lookup, regardless of
deterministic probe schedule. This property does not extend to arbitrarily
long reuse distances. Tag collisions can add candidates but cannot remove
a genuine recent repeat from this ring.

On a candidate, construct the full canonical key and compare complete bytes
before reuse; replay the selected policy against the original hypotheses.
On a miss, solve exactly, and independently check before admission in checked
arms. Coarse repeat alone can trigger unnecessary checking. The additional
confirmation arm requires a prior canonical digest observation in the
candidate stream before checking, still using full canonical bytes to
authorize hits. Digest hints never authenticate certificates. Trusted and
checked caches are distinct arms. The matched 384-byte logical allowance
covers packed histories; remaining 7808 bytes count serialized cache keys and
policies. This is not a Python object or RSS bound.

## Campaign

60 streams: six per cohort, 48 queries per stream, ten arms, two randomized
arm-order timing repetitions (57,600 arm-query executions). Smoke seeds
2026300000+, evaluation seeds 2026310000+100*cohort+ordinal; exact canonical
keys disjoint across evaluation cohorts. No fitting or calibration. Reuse
RV22-Q's exact solver, verifier, canonicalizer, and baseline state machine.
Eight prior workload families plus hidden reuse (row/test permutations,
duplicate splitting, weight scaling and insertion of irrelevant tests) and
collision (distinct random joint distributions with identical class masses
and chosen test marginal). Collision is constructed stress, not an estimate
of real frequency. CPU process time includes fingerprints, key generation,
solving, checking, replay and cache operations. No offline coverage audit in
timed region. Paired bootstrap 10,000 resamples over six streams, after averaging
the two repetitions. Small exploratory sample, no multiplicity adjustment.

## Coverage

Over 2880 query positions (before timing repeats), exactly 1447 had a canonical
repeat within the last 12 positions.

| Filter | True positives | False positives | False negatives |
|---|---:|---:|---:|
| Mass | 1447 | 318 | 0 |
| Single-test marginal | 1447 | 282 | 0 |

All 282 marginal false positives were deliberately constructed collision
queries. On the other nine cohorts mass has 36 false positives and marginal
has zero. This is empirical selectivity, not a universal collision bound.
The alias cohort has 210 recent repeats and zero missed by either filter.

## CPU results

Mean milliseconds per 48-query stream:

| Cohort | Cold | Trusted LRU | Checked LRU | Always second | Jitter1 | Checked mass | Checked marginal | Marginal + confirmation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Burst | 37.605 | 6.645 | 11.483 | 20.303 | 17.431 | 15.940 | 16.274 | 21.803 |
| Mixed | 32.349 | 9.635 | 17.258 | 22.757 | 28.078 | 24.756 | 24.274 | 26.608 |
| Churn | 39.020 | 38.612 | 77.668 | 40.512 | 39.452 | 39.474 | 39.598 | 40.615 |
| Phase | 33.683 | 23.254 | 44.287 | 27.310 | 27.965 | 27.556 | 26.616 | 28.730 |
| Recovery | 40.338 | 23.792 | 44.741 | 29.602 | 33.047 | 29.043 | 28.176 | 31.374 |
| Scan | 35.240 | 26.610 | 50.740 | 30.849 | 36.242 | 30.838 | 30.070 | 41.595 |
| Delayed | 34.572 | 13.226 | 23.900 | 26.887 | 35.410 | 28.027 | 27.584 | 35.353 |
| Alias | 53.473 | 13.179 | 23.940 | 16.152 | 20.955 | 15.668 | 15.649 | 17.413 |
| Hidden | 35.097 | 6.092 | 9.532 | 14.451 | 21.135 | 13.373 | 13.578 | 18.233 |
| Collision | 16.178 | 18.016 | 33.771 | 17.448 | 17.123 | 33.984 | 33.531 | 18.542 |

Checked marginal is faster than jitter1 with paired 95% intervals below zero
on eight cohorts. Against always-key delayed admission, only recovery
establishes a positive speed gain (-1.426 ms [-2.195,-.780]); mixed is slower
(+1.516 [.874,2.031]). Churn marginal constructs 1.833 keys per stream (96.2%
bypass) and checks 1.167; against cold +.578 [-2.837,3.174] ms, inconclusive.
On alias marginal constructs 35 keys and achieves 34 hits, one check;
vs jitter1 -5.307 [-7.993,-2.875] ms, vs always-second -.503
[-1.673,.411] ms. In hidden reuse, 38 hits and five checks.
Stable burst/mixed/hidden workloads favor ordinary full-key LRU in their
respective trust modes. Trusted marginal is slower than trusted LRU on
eight means and its apparent churn advantage is inconclusive.

The collision cohort has 47 candidates, zero hits, 47 checks per stream for
either unconfirmed checked filter. Marginal confirmation eliminates all
47 unnecessary checks, but adds an extra observation delay on useful roots:
burst 30 vs 36 hits, delayed 10 vs 20, hidden 33 vs 38. Collision confirmation
is still +2.364 [1.588,3.205] ms over cold. No universal speed advantage or
promotion. No adversarial availability guarantee. Use the cheap static cache
or existing delayed-admission control unless the workload and costs justify
a prefilter.

## Verification

All 57,600 arm-query executions agree exactly on utility. Two local regression
functions passed, including 24 independent index/used-test oracle comparisons
and checks, forced fingerprint-tag collisions, all-arm parity, irrelevant test
insertion, recent-repeat coverage, and zero-payload fallback. Source reuses
RV22-P/Q fixture code; it assumes generated valid input, and is not a production
certificate importer. Full numerical receipt is kept locally.

Reproduce:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/fingerprint.py --out /tmp/rv22r.json

Measured source SHA256 f81469353925e742defd6e1382485f6aa7d02c4f7814885bc635c73fefd44470.
Git blob 5a61096031c68c230704e8b583c5e7504c321415.
Test Git blob 7578e0912736a1e65fed1eb1a98c55c52b72f8b6.
Receipt SHA256 ff118e4d7b6702149b2d12416d6fb511eed10832e725bf64e20ba6c925999102.

## Next

RV22-S: distinguish cheap necessary invariants from sufficient canonical
identity. A lazy exact-key prefilter can use a stronger, incrementally
refined marginal or pairwise sketch when coarse ambiguity is observed,
while storing verified exact keys separately. Compare against optimized
always-key delayed admission and measure the additional fingerprint cost.
A theoretical collision lower bound for any fixed low-order marginal
projection is useful here; parity distributions provide witnesses.
Do not train another learned switch before this analytic comparison.
