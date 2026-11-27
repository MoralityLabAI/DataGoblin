# ALG-DESCENT-19: live goal-support filtering (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/674c360894269235dc5d38da61f5cbdc8f63293e). Synthetic public weighted acyclic Horn planning only. Fresh 12 IID roots 1410000..1410011, 24 OOD roots 1420000..1420023, same size/family schedule as V18. Four native arms: unrestricted packed hmax, static goal ancestor support, live support on transitions only, live support on transitions/hmax/FF. V18 search semantics unchanged except safe restriction of legal successor heads and heuristic computation. Exact full proved-set keys retained, no quotienting or learning. Native live cone descends topological heads using union of all prerequisites; stops at known facts. Arm live-all caches the cone per exact state. Shared static ancestor computation in bank load is outside timed region; live index initialization charged. All arms share new node layout and counters. Positive costs, exact rule registry, verified FF incumbent and immutable proof paths, best-g dominance/reopen, deterministic priority and complete unresolved improving-proof coverage.

## Results

36 graphs, 144 root/arm cases, 101 randomized-order repetitions each = 14,544 timed executions. All 144 certify exact optimum by 128 expansions. OOD means across 24 roots; latency root-average medians (microseconds).

| Arm | Expansions | Actual transition scans | hmax scans | FF scans | Live index scans | Total rule scans | Time us | Peak requested B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 14.8333 | 417.833 | 1429.083 | 419.292 | 0 | 2266.208 | 7.34225 | 7730.667 |
| static | 14.125 | 349.917 | 1061.417 | 351.375 | 0 | 1762.708 | 6.11633 | 6406.667 |
| live transitions | 14.125 | 348.833 | 1228.375 | 399.292 | 35.5 | 2012 | 6.84575 | 6406.667 |
| live all | 14.125 | 348.833 | 1039.583 | 350.292 | 35.5 | 1774.208 | 6.86621 | 6406.667 |

Primary live-all gate FAIL. Counted scans decrease 21.71%, 17 wins/7 losses; mean delta -492, descriptive root-bootstrap 95% [-840.04,-206]. Latency decreases 6.48%, 7 wins/17 losses; mean -476.04 ns, interval [-1287.5,+198.67] ns. Static ablation is strongest: scans -22.22%, 17 wins/7 ties, interval [-837.04,-230.42]; latency -16.70%, 23 wins/1 tie, delta -1225.92 ns, interval [-2070.38,-539.42]. Static is faster than either live arm on all 24 OOD roots in this run. Dynamic cone construction mean visited heads 139.25 transition-only and 414.54 live-all; this additional work is outside the counted rule-inspection sum and included in timings. All three support arms have identical mean expansions, 4 wins/20 ties versus full. Mean peak requested allocation -17.13% for all support arms, 5 wins/19 ties; maximum 11904 B vs 23168 B. No physical memory cap. The full baseline uses common V19 layout/instrumentation, so do not compare its absolute latency with prior cohorts.

Mean normalized certified gap at matched counted-inspection budgets (256/1024/4096):

| Arm | 256 | 1024 | 4096 |
|---|---:|---:|---:|
| full | .457293 | .197737 | .006290 |
| static | .446950 | .153891 | .001603 |
| live transitions | .457293 | .180427 | .006290 |
| live all | .449554 | .155494 | .001603 |

At 1024 inspections static and live-all improve gap on 7/24 roots, tie 17. Both certify 6/24 vs full 5/24. At 4096 both certify 23/24 vs full 21/24. Latest completed-event snapshots, not equal actual time; actual rule scans include hmax, FF, transition, live index, but exclude bit operations, visited-cone heads, proof checks, allocations, comparisons and other work. Separate support-head meter. Timing includes index construction, search, allocation/destruction and normal event digest/snapshot instrumentation; excludes common bank load, trace-only work snapshots, output and independent replay. Native trusted generated bank only.

## Correctness and audit

Independent set-based full Bellman oracle enumerates every full-reachable state (stop on goal). 48,617 states, 97,234 static/live filtered optimum comparisons, 180,896 cone-monotonicity checks. All agree. Independently recursively compute support from rule table, verify native exact choice, full hmax, direct FF completion, proof binding, best-g dominance and priority, frontier L/U and counters. 5,661 coarse bounds, 1,741 expansions, 7,402 events, 864 expansion snapshots, 432 work snapshots, 44,064 fields. All agree. Timing medians recomputed from raw samples; all non-timing repeated execution receipts and trace replay agree. ASan/UBSan with leak detection matches every algorithmic receipt and requested-allocation counter. PMR measures live requested search container bytes, excludes stack, allocator metadata, bank and output. Whole process max RSS 1792 KiB, not per-arm RSS. ABI Node 56 B, Residual 64 B (unused), Item 20 B, Engine 3408 B, Graph 460 B, Track 40 B, Result 2416 B.

## Retrospective quotient diagnostic

After the registered run, on the same spent graphs, group full-reachable states by root-local live cone. Compare exact remaining optimal cost and complete ordered live action signatures (original rule ID, head, cost, successor cone); all group members agree. 48,617 states -> 11,069 classes overall, 235,222 strict live-rule cone edges checked; 40,449 states have a strictly smaller live cone than their set of unknown facts. OOD aggregate 47,307 -> 10,294 classes (78.24% reduction); root-average fraction removed 71.78%. These are full state-space counts including terminal states, not observed OPEN sizes or memory reductions, and do not benchmark quotient search. The theorem in MathCan HORN_LIVE_QUOTIENT.md proves a sufficient exact key under the task's assumptions. Keep concrete witness paths and best g per class; scope immutable rules and goal. Next register a native quotient search against static filtering, measure cone computation plus actual memory/time, and test witness transfer under cheaper reopening. No learned control until stronger exact baseline is measured.

## Reproduction

From V19 directory with pinned generated V18 native source (v18.cpp, blob 6fd51b2cf3c759396a4fb57d96c9fa105a6d79d6) and V11 proof_planning.py (sibling alg_descent_v11 or prior/ in replay archive):
    python generate19.py v18.cpp native19.cpp
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python export19.py
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic native19.cpp -o native19
    ./native19 bank19.txt 101 > native19.txt
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer native19.cpp -o native19_san
    ASAN_OPTIONS=detect_leaks=1 ./native19_san bank19.txt 1 > sanitized19.txt
    python audit19.py
    python quotient19.py
Assertions enabled; Python/PyTorch import dependency. Source blobs match executed files: generate19.py 005d0fa554b77c4162739184d3b7fdc48f557d76; export19.py 80940942a6ddb01f81a7fe437dee01d9e4253c3a; audit19.py 2d5f4b26b263fee4d1904be4b2e16850850f2cc3; quotient19.py 51b76f225e86b078fbf79867c3b9aff2152580d4. Generated native19.cpp a611e8d94c51066ed5e4115bec20fadc78dd5e06. Artifact SHA256 bank 250580fdadf5dc3532391db0133008e5f5a688677a1974f6b623973dde3e6818; native e1ce1a4337697e60b38d15ecbf8ebc92842b5408e6fd767d1fa66f1b634c7f04; sanitized 7eeba021eedb91145ea965cdf0c0f4abcc37092070a2637fbb085cd54bb0c54b; audit b60d50e42caf64a7a2defb80b42f44f690cdc97eb16bbd77a9d5ee4e1f318c55; quotient 814f1019f81120956540be18a3ca769a7762f203a33e7f366b52499eb9185072. Timing hashes run-specific. No native BPD/MMS, Lean, external generalization or cryptographic claim.