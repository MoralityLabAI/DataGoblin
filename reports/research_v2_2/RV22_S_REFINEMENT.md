# RV22-S: marginal compactification bound and packed exact keys

Completed CPU-only exploratory experiment; no GPU lease, paid inference, or
controller promotion. Parent RV22-R general tests passed (800 passed / 15
skipped) and the dedicated verifier workflow passed; general CI failed on
a single SIM102 nested conditional, corrected in this patch.

## Mathematical bound

Fix m legal binary tests, their IDs and costs, rewards and budget. The
unnormalized joint distribution of class C in {A,B} and outcomes x in
{0,1}^m has 2^(m+1) coordinates. All class-conditioned marginals of order
at most k, interpreted as joint class/outcome masses, determine exactly
the Walsh coefficients of degrees 0 through k for each class. This follows
by expanding each intersection indicator as a product of (1 +/- chi_j)/2,
and conversely expanding each Walsh character as a product of (1-2 x_j).
Thus the linear map has rank 2 sum_{j=0}^k binom(m,j), and kernel dimension
2 sum_{j=k+1}^m binom(m,j). Normalization removes one independent scalar
from domain and image and leaves kernel dimension unchanged. These are
linear dimensions, not Shannon information or storage-size lower bounds.
For m=8, order 1/2/3/4 retains 18/74/186/326 out of 512 coefficients.

A concrete collision: use uniform outcomes and either independent balanced
class, or class determined by the parity of all m outcomes. For every proper
subset of outcomes, the joint class/observed-outcome masses agree (with the
same scale). The first distribution assigns mass 1 to each (class,x);
the second assigns mass 2 to the single parity-selected class for each x.
For k<m the normalized marginal signatures therefore agree, although
the exact joint distributions differ. With unit test costs, budget m,
correct reward 12 and wrong reward -16, stopping without revealing all
bits has best expected reward zero (abstain). Independent class remains
unpredictable after all tests, so optimal utility is zero. For parity and
1 <= m <= 11, testing all bits and predicting gives utility 12-m; any
policy that stops on some paths yields at most this value since a complete
path costs m and incomplete paths have nonpositive expected net reward.
Exact DP, replay and independent Bellman checks verified m=1 through 5.
For these five pairs, every marginal through m-1 agrees, the full-order
signature differs, and the optimal utility differs. Consequently equal
fixed-order signatures cannot authorize certificate reuse in general.

A packed exact key uses an integer code with class at bit zero and sorted
legal test outcomes above it. Aggregate integer mass by code, divide by
their GCD, and serialize sorted code/mass pairs together with budget,
rewards, and sorted legal test IDs/costs. For the valid fixture domain
this is a bijective re-encoding of the prior normalized canonical joint
distribution: unique fixed-length bit vectors decode from each integer
given the legal-test list. No truncation or digest equality is used for
cache authorization. Scale and exact key-equivalence parity are checked
over the complete evaluation inventory.

## Campaign

42 fresh streams, six per cohort, 48 queries per stream, eight arms and two
randomized arm-order repetitions = 32,256 arm-query executions. Smoke seed
2026410000, evaluation seed 2026420000+100*cohort+ordinal. Cohorts burst,
mixed, churn, deterministic-schedule alias, hidden equivalent roots,
coarse collision and pair_alias. Pair_alias uses four legal tests and
positive base 100000 with random Walsh perturbations of degrees 3 and 4.
All order-2 signatures agree despite distinct exact keys. Stress cohorts
are constructed, not estimates of real prevalence. No fitting or
threshold calibration. Cross-cohort exact inventories are disjoint.

All non-cold arms use an 8192-byte logical cap including 1024-byte reserved
metadata (three bounded hint rings and up to 16 coarse-bucket entries).
This is serialized/logical accounting, not a Python object or RSS bound.
JSON and packed second-observation controls construct exact keys on every
query and require a recent digest repeat before checked admission. Packed
trusted LRU is an eager, unchecked-admission control; do not equate its
trust mode with checked arms. Coarse marginal eager and packed marginal
eager construct exact keys only on coarse repeats. Ambiguity guard flags
a bucket after distinct exact-key digest observations and then requires
exact-repeat evidence for admission. Progressive pair additionally asks
for a repeated all-pairs signature in ambiguous buckets before exact-key
construction. The first ambiguous-bucket admission is not retroactively
removed. Digest collisions may cause extra work; they cannot authorize
a hit. Full key equality and policy replay remain mandatory. All checked
admissions independently check Bellman records. Pair signature hashes are
local hints; PYTHONHASHSEED=0 used for reproduction.

CPU process time includes fingerprint, pair moment, key, cache, replay,
solve and check operations, excludes offline inventory audit and bootstrap.
Pairwise 95% percentile bootstrap uses 10000 paired resamples over six
streams after averaging two timing repetitions. Small exploratory sample,
no multiplicity adjustment.

## Results

All 32256 executions agree exactly on integer utility numerators and
denominators. The 1060 distinct old/new keys are in one-to-one correspondence
with equal GCD scales. Total serialized key bytes: JSON 511031, packed
245682, a 51.924% reduction. This does not measure policy or metadata bytes.

Mean CPU ms per 48-query stream:

| Cohort | Cold | JSON second | Packed second | Packed trusted | Marginal eager | Packed marginal | Guard | Pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Burst | 37.909 | 17.747 | 18.064 | 6.717 | 17.557 | 17.656 | 20.611 | 17.139 |
| Mixed | 33.674 | 22.016 | 23.039 | 10.718 | 23.328 | 23.596 | 24.021 | 23.853 |
| Churn | 37.219 | 39.610 | 39.054 | 38.952 | 37.662 | 38.161 | 37.798 | 37.323 |
| Alias | 53.088 | 15.705 | 15.067 | 12.572 | 15.866 | 15.263 | 15.503 | 15.193 |
| Hidden | 34.148 | 13.161 | 12.816 | 5.778 | 13.216 | 13.187 | 13.182 | 13.194 |
| Collision | 16.103 | 17.938 | 17.851 | 17.944 | 34.327 | 34.737 | 24.151 | 18.466 |
| Pair alias | 16.468 | 19.672 | 19.380 | 19.894 | 38.114 | 36.800 | 19.942 | 23.067 |

Packed delayed and JSON delayed have identical hit and check counts across
these cohorts. Packed delayed minus JSON delayed CPU on churn -0.556 ms
[-1.127,-0.036], alias -0.638 [-1.438,-0.089]. Other cohort intervals
include zero. Compact storage is the robust observed advantage; broad
runtime superiority is not established.

On collision, the guard cuts checked admissions from 47 to 1 but still
constructs 47 keys. Pair refinement constructs 2 keys, probes pair moments
45 times, and checks once, with zero hits. Pair vs guard -5.685 ms
[-17.483,0.679]; pair vs cold +2.363 [1.620,3.047]. Against pair_alias,
the guard checks once, while pair refinement constructs 46 keys, computes
45 pair signatures and checks once. Pair vs guard +3.126 [2.275,4.472];
pair vs cold +6.600 [5.632,7.800]. Guard vs packed marginal -16.858
[-17.149,-16.593] on pair_alias. Both approximate screening strategies
remain costlier than cold on this adversarial cohort. No learned switch
training and no universal speed claim. Packed second is the simplest
checked candidate; guarded admission addresses coarse ambiguity at
small metadata cost, and pair refinement adds work without established
benefit relative to packed second. Keep both experimental.

## Verification and reproduction

Two local pytest functions passed, including five exact parity witness
pairs, permutation/scale/irrelevant-test and duplicate-hypothesis
invariance, all-arm exact utility across seven smaller streams, packed
bijection, forced coarse and exact digest collisions, and zero-payload
fallback. Valid synthetic fixture input is assumed. This is not a production
certificate parser. The source used for timing differs from the committed
source only by removal of unused H import and a docstring clarification;
the final source passed local regression again. Hosted CI for RV22-S
must be checked separately.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 python scripts/research_v2_2/refinement.py --out /tmp/rv22s.json

Measured source SHA256 d4209396b8f878917d7732753dab335edf18032bf9c60f60150d0a8760ed1eb5.
Final source SHA256 64eb6052f51ceb814c67a7c5416a52f81f03fef06edc4a580c89d24de517ab36.
Final Git blob 3c42d8a2b1175d1fa8b51532a1f49f1c53234896.
Test Git blob 67fe453ccc5aa206b2e7bb31f94ad42db5d2588c.
Receipt SHA256 16c61d7070761fb9c80fda14f5e1c5004263d46661d96520d6ee691dd3e05291.

## Next

RV22-T: investigate compact exact-key capacity curves using the same
checked admission semantics. Vary logical payload capacity, reuse
distance and joint distribution sparsity; compare integer-coded JSON
against a versioned binary encoding, with independent decoding and
round-trip/collision tests. Audit parser bounds and metadata overhead.
Separate storage wins from CPU wins; do not train a learned switch
until an independently confirmed utility/CPU advantage is measurable.
