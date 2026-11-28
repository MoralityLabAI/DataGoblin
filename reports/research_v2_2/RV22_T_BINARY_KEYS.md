# RV22-T: bounded binary canonical keys and cache capacity curves

## Status and contract

CPU-only synthetic experiment. No paid inference, GPU use or promotion.
Parent RV22-S commit c092b31a passed both hosted workflows.

A binary re-encoding of the RV22-S integer-coded normalized joint
distribution uses versioned magic DGK/1, a big-endian fixed header, UTF-8
test IDs with explicit lengths, uint32 test costs, sorted integer outcome
codes and minimal common unsigned mass width. The header carries budget,
signed rewards, test count, row count and mass width. The class occupies
bit zero of the code. Sorted legal tests determine all other bit positions.
The decoder rejects bad versions, length mismatch, invalid UTF-8,
duplicate/unsorted/reserved IDs, nonpositive or unaffordable costs,
constant legal tests, out-of-range or duplicate/unsorted row codes,
zero or non-GCD-normalized masses, and nonminimal mass width.
Bounds: 32 legal tests, 4096 rows, 128 UTF-8 bytes per test ID, 64 KiB key,
four bytes per mass. The encoder accepts trusted fixture records; it is
not a validator for arbitrary external world inputs. This is a key codec,
not a production certificate importer or an authentication mechanism.

Every checked arm uses identical delayed admission and LRU replacement:
64-query digest repeat hints; full key equality for hits; exact policy
replay; independently checked Bellman records for all admissions. Policy
payload encoding remains unchanged. Fixed accounting reserve 1024 bytes
for up to 64 digest/time history records. Capacities 2/4/8/16 KiB refer
to serialized key and policy bytes plus the logical reserve, not Python
RSS. Retained-object estimates are recorded separately; temporary solver
allocations, allocator overhead and other process memory are excluded.

## Protocol

Cyclic recurring root sets at exact reuse distances 4, 12 and 24. Each
query uses the prior variant generator (permutations, mass scaling and
duplicate splitting). Sparse8 uses the prior 16-hypothesis random
eight-test generator; dense4 has all 32 class/outcome combinations with
positive random masses and four tests. They differ in problem size as
well as sparsity; no isolated sparsity causal claim.

Initial expanded grid was interrupted during the sparse8/distance24
cohort by the execution timeout. Retain only the two completed cohorts
(sparse8 distances 4 and 12, four streams each, 72 queries, two timing
repetitions, base seed 2026550000). Complete a fresh smaller grid across
all six cohorts at seed 2026560000, two streams each, 72 queries and
one timing repetition. Each cohort uses the same four arms and four
capacities, with shuffled arm/cap order. Never pool the two protocols.
The full fresh grid has 13,824 arm-query executions and the expanded
completed cohorts have 18,432, total 32,256. All agree exactly on integer
utility numerator and denominator across all arms and capacities.
The 224 distinct canonical root digests across completed streams are
disjoint between streams and cohorts. All query-level round trips,
GCD scales and one-to-one old/new key mappings checked; the integer
JSON representation matches RV22-S packed keys.

Process CPU time charges key construction, solve, checking, replay,
lookup and eviction; excludes the offline key audit, retained-object
estimation and paired bootstrap. Timing measurements on this CPU are
exploratory. The two-stream grid is particularly small and includes
timing outliers; its stored bootstrap percentiles should not be read
as strong inferential evidence. No learned component or tuning.

## Storage result

On the fresh grid's 160 distinct exact roots, key bytes total:
original JSON 89,850; integer JSON 37,853; binary 18,995.
Binary reduction 78.9% vs original JSON, 49.8% vs integer JSON.
These are key bytes only, not whole-cache or resident-memory savings.

Smallest tested logical capacity that achieves all post-second-observation
hits on both streams (KiB):

| Profile / reuse distance | Original JSON | Integer JSON | Binary | Hits/72 |
|---|---:|---:|---:|---:|
| sparse8 / 4 | 4 | 4 | 2 | 64 |
| sparse8 / 12 | 16 | 8 | 4 | 48 |
| sparse8 / 24 | 16 | 16 | 8 | 24 |
| dense4 / 4 | 4 | 4 | 2 | 64 |
| dense4 / 12 | 16 | 8 | 4 | 48 |
| dense4 / 24 | 16 | 8 | 4 | 24 |

This is a capacity cliff specific to cyclic LRU: below the working-set
size a repeat can systematically miss. Above it the cost of two
observations per root remains. No promotion or universal CPU advantage.

Expanded completed cohorts: CPU ms per stream (mean of four streams,
averaged over two timing repeats).

| Profile, cap | Cold | JSON | Integer JSON | Binary | Binary hits |
|---|---:|---:|---:|---:|---:|
| sparse8/4, 2 KiB | 103.26 | 191.67 | 190.12 | 20.13 | 64 |
| sparse8/4, 4 KiB | 98.51 | 21.02 | 19.90 | 19.58 | 64 |
| sparse8/4, 8 KiB | 96.36 | 20.47 | 19.98 | 19.74 | 64 |
| sparse8/4, 16 KiB | 97.53 | 20.66 | 19.61 | 19.74 | 64 |
| sparse8/12, 2 KiB | 90.26 | 168.57 | 166.95 | 169.21 | 0 |
| sparse8/12, 4 KiB | 90.03 | 164.25 | 161.86 | 46.93 | 48 |
| sparse8/12, 8 KiB | 89.23 | 78.56 | 48.44 | 50.05 | 48 |
| sparse8/12, 16 KiB | 87.00 | 48.94 | 49.21 | 47.50 | 48 |

At sparse8/4 2 KiB, binary vs integer JSON paired CPU delta -170.00
ms, percentile interval [-184.04,-160.80]; at sparse8/12 4 KiB
-114.93 [-119.66,-110.21]. Both depend on recovering hits; with all
entries resident the differences are small and inconsistent. Fresh
grid long-distance dense4/24 at 4 KiB: binary gets 24 hits but takes
22.94 ms vs cold 20.73 ms, so fit alone does not guarantee a speedup.
The cold baseline is repeated across capacities solely to match timing
order; capacity has no effect on its algorithm.

## Verification and reproduction

Three local pytest tests passed: round trips and reconstructed key parity,
hidden variants, parity witnesses, exact solver/replay/check consistency,
malformed/truncated/noncanonical keys, forced digest collision with full
key authorization, zero payload and checked capacity accounting.
The timing source was line-wrapped once without semantic change.

Measured source SHA256 1dca8d2e0cf9effcef8320e4d74339b137971cc0459594d0a727ac36f34151da.
Final source Git blob 4ac15e8c69fed2cb965e6152005e84ea15803b98.
Test Git blob 05949aab71605509dece183d33f56f6926895666.
Combined local receipt SHA256 fa2e19fc26c0aae67b5cf04395c277951a2e8d7b39bb0547e04d653b7ad90b74.

Reproduce the small grid with separate fresh output files for each
profile in sparse8/dense4 and distance in 4/12/24:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/binary_keys.py --profile sparse8 --distance 4 --streams 2 --length 72 --repeats 1 --seed 2026560000 --out /tmp/rv22t_sparse8_4.json

Expanded protocol uses streams=4, repeats=2, seed=2026550000.
Hosted CI for RV22-T has not yet been checked.

## Next

RV22-U: compare eager and delayed exact-cache admission under working-set
overflow using measured retained-byte and certificate-check costs.
Test admission that avoids repeated checks for scan-resistant entries,
and distinguish resident capacity from key-encoding improvements.
Freeze the codec and preserve exact replay. No learned switch yet.
