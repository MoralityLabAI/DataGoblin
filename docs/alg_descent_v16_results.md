# ALG-DESCENT-16: packed native Horn certification (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/98d9b55d81716f29beee177ce4979bb22d2c270f). Fresh V15-generator public synthetic acyclic Horn graphs: 12 IID roots 1110000..1110011, 24 OOD 1120000..1120023. C++17 release and ASan/UBSan builds. No learning, factoring, DLP or hidden solution witness.

Native hmax, scan-lazy and indexed-lazy use the same search, exact verifier, verified FF incumbent/completions, state dominance, deterministic priority and complete unresolved OPEN. The cut arms preserve V15's single-round refinement. Indexed uses uint64 rule masks, exact zone memoization and affected-descendant distance updates; no transferred cuts. Immutable paths are byte rule IDs, residual costs and distances are byte arrays. n<=18, <=45 rules; intended for the trusted generated graph bank with 1..3 earlier premises per rule, positive costs 1..7, fixed initial facts and last-fact goal. This is not a hardened arbitrary-bank parser or a portable authenticated certificate service. No external solver values enter native search.

## Results

36 roots x 3 arms x 61 randomized-order repetitions = 6,588 timed release executions. Trace-off timing includes search initialization, index construction, snapshot/hash instrumentation, exact incumbent checks, allocator accounting and search destruction. Excludes parsing the shared graph bank, trace vector construction/serialization, offline reference and independent audit. Timed repetitions agree on all non-timing result counters and event digest; full trace-on execution per case agrees. Only per-root timing medians are persisted, not raw timing samples. GCC 14.2.0 -O3, AMD EPYC 9V74 host, no CPU pinning or general timing claim.

OOD root means:

| Arm | Capped expansions | Bound-rule inspections | Mean root-median time (us) | Certified at 128 |
|---|---:|---:|---:|---:|
| hmax | 16.9583 | 1815.9583 | 13.05346 | 23/24 |
| scan-lazy | 12.0833 | 7641.2083 | 26.65129 | 23/24 |
| indexed-lazy | 12.0833 | 3374.4167 | 25.23608 | 23/24 |

All 108 cases return verified incumbents; 105 certify optimality by 128, including all IID cases. All three arms remain unresolved at root 1120017 (optimum 31): hmax [27,31], cut arms [29,31]. Capped expansion means include the unfinished searches. OOD indexed vs scan inspections -55.84%; time -5.31%, mean paired -1.4152 us (16 wins/8 losses, descriptive root-bootstrap 95% interval [-3.3153,-0.3081] us). Primary gate FAIL (required >=10% time saving). Secondary gate vs hmax FAIL despite 28.75% fewer capped expansions; indexed takes about 93.3% longer, with same number of certificates. Four indexed wins against hmax on elapsed time. No retuning. Rule inspections count distance/boundary/minimum rule visits, exclude incidence bit operations and do not equal machine instructions.

Mean normalized interval widths at expansion budgets 0/4/16/32/64/128:
hmax .390720/.146199/.037215/.015655/.008065/.005376;
scan/index .390720/.108444/.020557/.010515/.006720/.002688.
The tighter bounds reduce the certified gap at equal expansion count; equal expansion count is not equal elapsed compute.

## Measured storage

All five dynamically allocated search containers (nodes, residuals, exact state hash table, heap, zone table) use the tracking PMR resource. Peak live requested bytes (includes simultaneously live old/new vector capacities during growth), requested-byte totals, and allocation counts are measured; deallocation returns live count to zero on every run. This does not include allocator bookkeeping/rounding, stack, external graph bank or benchmark/trace/output allocations. Residual vector populated only by cut arms. Object sizes on tested ABI: Node 48 B, Residual 64 B, queue Item 20 B, Engine 2224 B (includes embedded Result), Graph 460 B, Track 40 B, Result 1384 B. Object sizes are not measured whole-stack peaks; do not sum embedded objects twice.

| Arm | Mean peak requested B | Max peak requested B | Mean total requested B | Mean allocation requests |
|---|---:|---:|---:|---:|
| hmax | 8972.33 | 48136 | 12879.67 | 80.71 |
| scan-lazy | 11131.33 | 85016 | 17538.33 | 65.21 |
| indexed-lazy | 11865.33 | 90888 | 18632.67 | 91.92 |

Indexed uses fewer search states on average (mean peak best table 48.21 vs hmax 66.96) but more requested memory because of residual/index storage. Whole release-process ru_maxrss 1920 KiB, a coarse shared process-level high water including other program allocations, not arm-level RSS or PSS. No physical memory-cap claim. Instrumentation may influence microsecond timings. No independent allocator/stack/RSS measurement.

## Independent verification

Python reference uses the V15 implementation with seven repeated executions, identical non-timing receipts. Independent V15 set-based audit on all 36 new graphs: 53,970 exact Bellman states, 4,425 bound certificates, 4,716 cuts by deletion/unreachability, 9,141 bound prefixes, 5,867 priority events (1,151 expansions and 4,716 refinements), 648 snapshots, 20,088 fields and 3,426 incumbent proof steps. Independent full residual distance comparisons cover 39,992 coordinates. Native matches reference on all 108 full ordered search traces (5,867 events), all 4,425 coarse bounds and 4,716 ordered zone/delta/rule-mask cut records, 648 snapshot proofs and 18,792 snapshot fields including counters. Every interval contains offline optimum. ASan/UBSan with leak detection replays all 108 cases, identical algorithmic fields, event hashes and tracked allocation counts, no sanitizer report. Native algorithmic replay inherits reference verification via exact matching; native internal residual vectors are not separately exported and audited coordinate by coordinate. Native input bank fixed by hash; replay does not authenticate arbitrary external inputs. No Lean formalization.

## Reproduce

Use the bundled replay ZIP or copy the pinned dependency files into a prior/ subdirectory beside the V16 scripts: V11 proof_planning.py, V12 anytime.py, V13 bounds.py, V15 experiment.py/indexed.py/audit.py. Requires Python with PyTorch import dependency; no model training. From that directory:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python reference.py
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic native.cpp -o native
    ./native bank.txt 61 > native.txt
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer native.cpp -o native_san
    ASAN_OPTIONS=detect_leaks=1 ./native_san bank.txt 1 > sanitized.txt
    python compare.py
Run Python with assertions enabled. Git blobs matched locally executed source: native.cpp 4137582b8c3b3f1c71245b5a00a88ecfa7205814; reference.py faf13de44e204e260fd15391d0fdb02178f5d2f4; compare.py 733f77efab6820b561ee2e1695d58a3bed169024.
Artifact SHA256: bank.txt 7638ecd7eb5c05a2b165ea76a57571d1b8973d8bec065c4592b56663ade70652; reference.json b0ed65e0ac531e3026e78704719c4c434970ea40af8124d19604aa891aa91ac2; native.txt e5286da78be9b086470979ed68b93baf78446dac9280aa77281e3ee0673d48f4; sanitized.txt 30c781af591381f9c26fd5a9f412b6d610326cceb30dd6dfd156a193bc7c6690; comparison.json 22d497d7beb1e707fc9258807a600e8e97dba06416b072a0371dd612da309643.

Next: compare exact direct bitset boundary computation without zone hash memoization, flat/sparse state indexing and deferred residual allocation (compute coarse hmax without retaining the vector until refinement), with memory and latency counters, equal trace and budget checks. Separate exact algorithm improvements from optional instrumentation removal; compare all arms using the same instrumentation. No native BPD/MMS or cryptographic/general proof-search claim.