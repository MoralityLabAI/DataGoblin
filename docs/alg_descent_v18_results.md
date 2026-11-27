# ALG-DESCENT-18: budget-aware certified refinement (2026-11-27)

Prospective protocol: [e4412c1](https://github.com/MoralityLabAI/DataGoblin/commit/e4412c1fadce730bea56aeb40d5bee07e60bf044). Synthetic public acyclic weighted Horn graphs only. No hidden witness, neural training or cryptographic discovery. Fresh 12 IID roots 1310000..1310011 and 24 OOD roots 1320000..1320023. Six policies share V17's packed state index, direct exact cut boundary and deferred residual storage (cut arms only), exact rule-ID proof verifier, verified FF incumbent/completions, best-g dominance, reopen, immutable proof paths, deterministic priority and complete OPEN. Expand with the current admissible bound when the refinement policy declines. Retain partial bounds and residual records on reopen. Per-state cut counts and slack U-g-b determine the policy. No eviction. Expansion snapshots 0/4/16/32/64/128; refinement cap 4096. All proofs remain within the native input/domain limits.

## Results

36 roots, six policies, 216 cases, 101 randomized-order repetitions/case, 21,816 timed executions. All 216 cases certify optimality by the expansion cap (maximum 89 expansions, 429 selected refinements). Mean per-root median times over 24 OOD roots:

| Policy | Expansions | Refinements | Inspections | Time us | Peak requested bytes | Certified |
|---|---:|---:|---:|---:|---:|---:|
| packed hmax | 24.875 | 0 | 3248.417 | 12.54725 | 11108.50 | 24/24 |
| full cuts | 18.333 | 111.583 | 5863.333 | 26.03542 | 11374.50 | 24/24 |
| one cut | 24.875 | 24.875 | 4194.125 | 16.86354 | 12780.50 | 24/24 |
| two cuts | 22.292 | 49.167 | 4415.042 | 18.90592 | 12254.83 | 24/24 |
| slack<=2 | 19.667 | 48.042 | 4084.792 | 16.94942 | 10410.50 | 24/24 |
| slack<=4, max two | 22.250 | 39.417 | 4121.750 | 17.24688 | 11844.17 | 24/24 |

Inspections = transition probes + distance, boundary and minimum scans. Excludes FF, proof checking, bit operations, queue/index overhead, allocation, and other primitive work. It is a partial work meter, not an equal-time budget. Primary gate FAIL: slack<=4/max two saves 10.55% expansions but costs 37.46% more time (2 timing wins/22 losses); fixed two-cut policy also fails. Slack<=2 saves 20.94% expansions (20 wins/4 ties), reduces mean peak requested bytes 6.28%, but costs 35.08% more time (6 wins/18 losses; paired difference +4.402 us, descriptive root-bootstrap interval [1.565,7.797]). Full cuts save 26.30% expansions and more than double elapsed time. One-cut expansion count ties hmax on every OOD root. No retrospective primary selection.

## Matched inspection snapshots

Normalized interval width (U-L)/offline optimum, mean OOD. Each snapshot uses the latest completed search event within the cap, or the final proof if search terminates within the cap; the next event is never partially counted. The native trace-on run collects these snapshots without changing policy; timing runs omit this extra work-snapshot recording for all arms.

| Policy | Gap at 256 | Gap at 1024 | Gap at 4096 | Certified at 1024 |
|---|---:|---:|---:|---:|
| hmax | .376267 | .174676 | .029794 | 3 |
| full | .346692 | .274269 | .072068 | 4 |
| one | .395978 | .222523 | .040654 | 1 |
| two | .385511 | .217610 | .043729 | 2 |
| slack2 | .362841 | .178389 | .029794 | 4 |
| slack4_two | .377740 | .190427 | .037740 | 2 |

Secondary gate FAIL at 1024. Slack4_two minus hmax mean gap +.015751, 1 win/17 ties/6 losses, descriptive bootstrap [-.001752,.034959]. Slack2 +.003713, 2 wins/19 ties/3 losses, interval [-.010813,.017992]. Every policy has a verified incumbent already at budget zero. Actual mean inspections used at 1024 differ (hmax 917.58, slack2 851.67, slack4_two 877.38, full 849.58) because of atomic search events and early termination. At 4096, hmax and slack2 certify 19/24 and have identical mean normalized gap. This is not equal actual time or physical RAM.

## Audit and memory

Independent Python replay reads the exact immutable public graph bank and uses sets for transitions, offline Bellman optimum, full topological residual distances and cut deletion reachability. Reconstructs every selected action, path, coarse bound, cut membership and charge, current bound, best-g and reopen, FF completion, pruning, priorities, heap counters and snapshots. 60,720 independently enumerated Bellman states; 13,326 coarse bounds; 7,444 unavoidable cuts and 123,853 full residual distance coordinates computed by the reference; 24,418 native events (including 3,648 expansions); 1,296 expansion snapshots, 648 matched-work snapshots and 60,264 compared fields. All assertions pass. The native residual vectors are not exported coordinate by coordinate; audit recomputes full distances, verifies unchanged coordinates outside the affected descendant region and checks emitted bound/cut records. Exact optimum contained in all reported intervals. All release receipts match ASan/UBSan full replay including allocation counters and event hashes. Timing medians recomputed from raw samples; sanitizer timings excluded. The bank is generated from the pinned V11 code and its deterministic hash is checked on regeneration; arbitrary untrusted bank parser hardening is outside scope.

Peak requested bytes are PMR live allocation accounting for search containers, excluding stack, allocator metadata, shared graph bank and trace/output; not a physical RAM budget. Layout Node 52 B, Residual 64 B, Item 20 B, Engine 3096 B, Graph 460 B, Track 40 B, Result 2200 B on tested ABI. Whole release process max RSS 1920 KiB, not per-arm private memory. Timed region includes index initialization, search, allocation/destruction and ordinary search snapshot/hash instrumentation, excludes parsing, work-snapshot trace recording and independent replay. Host-specific results, no timing or memory superiority claimed beyond measured comparisons. No Lean, native BPD or native MMS integration.

## Reproduce

From this experiment directory, use pinned V17 native.cpp (git blob ff9711045ac44bc5648edbb60e1979a06fdb8a2d; named v17.cpp in replay ZIP) and V11 proof_planning.py (blob 7741baa9fd0fce589d9228b7a085ec3ccfdbc05c). The exporter uses the sibling V11 directory or prior/ in the archive. With Python/PyTorch import dependency and C++17:
    python generate18.py v17.cpp native18.cpp
    python export18.py
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic native18.cpp -o native18
    ./native18 bank18.txt 101 > native18.txt
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer native18.cpp -o native18_san
    ASAN_OPTIONS=detect_leaks=1 ./native18_san bank18.txt 1 > sanitized18.txt
    python audit18.py
Run with assertions enabled. Source git blobs verified: generate18.py 5079c5c191124f454c771f73af411eb946c5d824; export18.py 5ad02be46f9a94877cf861fb68b59ff3a67fea87; audit18.py 8ffff32aaf288f38946c2d6fefea9810da3e2d0e. Generated native source blob 6fd51b2cf3c759396a4fb57d96c9fa105a6d79d6. SHA256 bank ed7ccc7f784f0f7e74222d07324def4a89df6c73c1d45344d3e8c25b03c42fa8; native receipt 6c2d7b8463f532398612189f8364808e586d7ef613731d5be56f39337150362b; sanitized fd7b39ef1d258a16721c2895f5ad1cde66a5ecd837059ead0ae3291445731981; audit 2395ac15fe9c033aad08ae46c83c632e875018ce9b2b38918ffc4aa578b74590. Timing-containing artifacts are run-specific.

Next: reduce the work of packed hmax itself using exact goal-support filtering and state-relative live ancestor cones. This is a correctness-preserving search restriction under the monotone positive-cost task assumptions, to be tested prospectively; it has not been benchmarked here. Keep admissible pruning and complete improving-proof coverage. Do not train a learned refinement policy until there is evidence of headroom over the optimized exact control.