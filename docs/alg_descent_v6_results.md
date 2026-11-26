# ALG-DESCENT-06: exact interned-state cache (2026-11-26)

Prospective protocol: [f6836f5](https://github.com/MoralityLabAI/DataGoblin/commit/f6836f562047b431aa7037bc4e7abaa725669a73). Synthetic public-witness polynomial certificate routing only. No factor discovery, DLP solving, neural training or native BPD/MMS integration. Fresh IID seeds 170000..170023 (degrees 8/12/16/20), OOD seeds 180000..180031 (24/32/40/48). Frozen v4 train-only empirical prior and symbolic risk rule, v5 public bank order. Native exact product verifier unchanged.

## Reproduce

From repo root (Linux C++17 and Python with v4 dependencies):

    SRC=src/datagoblin/research/alg_descent_v6
    D=/tmp/alg-descent-06
    mkdir -p "$D"
    python "$SRC/generate.py" "$D/native.cpp"
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic "$D/native.cpp" -o "$D/native"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/export_bank.py" "$D"
    "$D/native" "$D/bank.bin" 101 > "$D/native.csv" 2> "$D/native.log"
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$D/native.cpp" -o "$D/native_san"
    ASAN_OPTIONS=detect_leaks=1 "$D/native_san" "$D/bank.bin" 1 > "$D/sanitized.csv" 2> "$D/sanitized.log"
    python "$SRC/check_receipt.py" "$D" > "$D/audit.json"

All four v6 source files match the executed local git blobs:
generate.py 5758f8f8c4cf17349d342c61dc9fd5f11eec2d9c;
benchmark.inc 24602d2888a9d986abdf0b6476b9451a83410b89;
export_bank.py 178ae754b63b525523fb888f92c6e1bee9cda948;
check_receipt.py a4bc40c8ad4404446d08d042bf7d6aaa2aff439f.
Generated translation unit git blob f601bf72ffad52e021cfc6efb22d5f3eb86f7134.
Local raw artifact SHA256:
bank.bin c79ec4062b2d074f36a80ea85b97539b8670350a75f6f84b510102ec8746b999;
native.csv 8e81f1da7ce2b2f98491f382ebf735b1e078ba43cc4d5789603675ba2d2d1185;
reference.json f95c8e961638e95d8e72b36366485fad460ea940fdf21f624cb994f584d6acf9;
audit.json e9fc7b8e64104f1fcda3a1ce940ed7fa5300cccb120634776856ca743269458b.
Timing-containing artifacts have run-specific hashes. Raw artifacts supplied separately rather than committed.

## Exact identity and allocation

Sort root-local node indices by full canonical polynomial (length plus exact packed coefficients), breaking ties by original index. Equal keys have the same representative, the least original ID. Interned cache uses uint32 representative with empty sentinel, recency and saved work, 12 B per slot. Packed baseline 36 B. The 16-bit representative occupies offsets 26..27 in previously unused alignment padding of Node; sizeof LegacyNode and Node both 56 B, alignment 8, action-vector offset 32. Runtime asserts these sizes and rejects unsupported layouts rather than claiming zero incremental storage on other ABIs. Fixed controls 32 B, pending stack 104 B for both. At entry allocation 72/108 B, total accounted live cache+frontier storage is 208/244 B; allocator overhead, recursion, bank and benchmark allocations excluded. Equal-allocation budgets are exact multiples of both slot sizes. Matched-slot control uses one third of entry allocation, same policy. Packed baseline shares the modified node ABI and pre-interned bank but never consults representative IDs.

All 22,592 OOD bank nodes have 5,025 distinct root-local keys. Root-local unique counts match independent Python tuple equality. Temporary sort vector payload peaks at 2,352 B (plus allocator and sort stack); released before rollouts. If padding cannot be reused, a conservative persistent sidecar payload costs 600..2,352 B per OOD root, before overhead, and would largely consume these small budgets. No persistent deduplication of nodes or candidates. Key identity is safe for caching existence of a verified closure because the exact polynomial and field determine the certified factorization predicate; it does not identify alternative candidate banks or cost-to-go. Cache stores trusted in-process receipt flags only after full closure. No cross-root reuse, no probabilistic key equivalence, no arbitrary hostile-bank service claim.

## Results

56 fresh roots, 37 arms per root, 2,072 root/arm cases, 101 randomized-order native timing repetitions per case = 209,272 timed executions. All cases close under work cap 3*n^2 and call cap 8*n. Charge every attempted exact product and one unit per cache hit. Python tuple-key replay agrees on all 14,504 fields (closure, work, calls, rejects, hits, peak occupied slots, pending peak). All 672 equal-slot comparisons agree exactly. ASan/UBSan full single-repeat replay agrees on every non-timing field. Native inherited v5 malformed-certificate and packed-key selftests also pass.

OOD mean work and root-average median warmed time:

| Entry B | Packed LRU work | ID LRU work | Packed pending work | ID pending work | Packed LRU us | ID LRU us | Packed pending us | ID pending us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1031.25 | - | - | - | 3.043 | - | - | - |
| 36 | 933.78 | 904.72 | 933.78 | 880.03 | 2.764 | 2.562 | 2.793 | 2.567 |
| 72 | 921.31 | 864.53 | 897.66 | 857.47 | 2.719 | 2.410 | 2.675 | 2.503 |
| 108 | 904.72 | 850.50 | 880.03 | 848.03 | 2.643 | 2.371 | 2.615 | 2.446 |
| 252 | 857.16 | 844.66 | 852.66 | 844.66 | 2.435 | 2.456 | 2.509 | 2.456 |
| 504 | 846.53 | 844.66 | 846.53 | 844.66 | 2.447 | 2.721 | 2.463 | 2.718 |
| 1008 | 844.66 | 844.66 | 844.66 | 844.66 | 2.607 | 3.262 | 2.607 | 3.259 |

PASS registered corresponding-policy gate for LRU: at 72 B mean work difference -56.78125 (6.16%), wins/ties/losses 29/3/0, descriptive root-bootstrap 95% [-75.84375,-39.90625], mean median time difference -309.56 ns. At 108 B -54.21875 (5.99%), wins/ties/losses 27/5/0, interval [-75.40625,-35.65625], time -272.0 ns. Pending-aware corresponding-policy gate fails: only 36 B clears 5% (53.75 work, 5.76%; 29/3/0); 72 and 108 B improve 4.48% and 3.64%. Against strongest packed policy at each cap, best interned work improves 5.76%, 4.48%, 3.64% at 36/72/108 respectively, so no two-capacity >=5% superiority over strongest packed control. Equal-slot ID LRU has identical work and lower mean warmed median times in this run at all six capacities (e.g. 72 B packed 2.719 us vs equal-slot ID 2.665 us using 24 B). At larger ID capacities, linear scans of mostly empty slots increase latency; this is a plausible explanation, not a separately profiled causal claim. Best mean warmed OOD latency is ID LRU at 108 B, 2.371 us. Timings are host-specific and descriptive.

## Preparation costs

Native load 62.55 ms across 56 roots. Prior construction 0.401 s, Python evaluation bank generation 3.246 s, export 0.215 s, independent reference 0.215 s. Whole native process max RSS 23,552 KiB (~23.0 MiB), includes banks, allocator and benchmark. No physical RSS reduction established. OOD interning mean 39.861 us, median 35.122 us; measured vector allocation/sort/assignment only, excludes post-intern equality checks and temporary vector destruction. Approximate amortization using ratio of means at 72 B LRU: 39.861 us / 0.30956 us ~129 repeated warmed rollouts, and ~147 at 108 B. This is an illustrative lower estimate, not a measured cold break-even. One-pass speedup is not established. Shared bank generation dominates.

## Decision and next work

The exact ID representation is useful at tight entry allocations on this ABI; the accepted result is a scoped corresponding-LRU improvement. Further optimize active-slot occupancy, avoid scanning unoccupied entries, and test native bank preparation and exact representative-aware reuse across verified tasks. Preserve separate candidate-bank versions when merging nodes: closure equivalence does not imply route-cost equivalence. Reuse benefits must be amortized over repeated work. Risk prediction across held-out corruption families is a separate prospective experiment, still outstanding. No native BPD/MMS integration or general proof-search speedup demonstrated.
