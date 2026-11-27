# ALG-DESCENT-17: allocation-efficient native certification (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/92fbc8ab06cae41133fcd87fb7f5aa430068f69e). Synthetic public acyclic weighted Horn proof planning only. Fresh IID 12 roots 1210000..1210011, OOD 24 roots 1220000..1220023, V11/V15 generator and V16 native search. Full 2^3 factorial indexed-cut implementation ablations: direct incidence union (no zone memo), defer full residual allocation until first selected refinement, exact packed open-addressed index. Two additional hmax controls with old hash table or packed index. All arms share verified FF incumbent/completions, legal transitions, dominance, immutable paths, admissible pruning, identical lazy refinement, expansion/refinement caps and instrumentation. No learning, cryptographic task or hidden witness.

The index encodes s>>3 (15 bits, first three facts always known) in upper 16 bits, node ID+1 in lower 16 bits; zero denotes empty. Exact key comparisons, linear probing, power-of-two growth above 0.7 load, no deletion. Rehash preserves stable node IDs and search insertion serials. Deferred residual recomputes original distances on first refinement, checks coarse goal distance, then stores exact residual. Extra scans charged. Direct boundary union uses same incidence masks and rule order. None of these optimizations change admissible bounds or pruning semantics.

## Results

360 cases, 101 timed randomized-order repetitions each (36,360 timed executions; raw samples retained), separate trace-on replay and one sanitizer replay per case. All 360 certify optimum by 128 expansions. OOD root-average medians and requested storage:

| Arm | Mean median us | Mean peak requested B | Mean allocations | Residual entries | Bound scans |
|---|---:|---:|---:|---:|---:|
| hmax | 12.06158 | 8179.17 | 74.33 | 0 | 1524.96 |
| hmax packed | 9.12421 | 7119.17 | 15.54 | 0 | 1524.96 |
| indexed | 17.94108 | 11529.00 | 85.29 | 43.42 | 2672.50 |
| direct | 16.50638 | 10884.67 | 61.71 | 43.42 | 2672.50 |
| defer | 18.85538 | 7851.67 | 83.92 | 13.67 | 3055.88 |
| direct+defer | 17.42112 | 7180.00 | 60.33 | 13.67 | 3055.88 |
| packed | 15.53625 | 10720.33 | 43.08 | 43.42 | 2672.50 |
| direct+packed | 14.19088 | 10082.00 | 19.50 | 43.42 | 2672.50 |
| defer+packed | 16.47642 | 7016.33 | 41.71 | 13.67 | 3055.88 |
| all | 15.10767 | 6361.33 | 18.12 | 13.67 | 3055.88 |

Primary gate PASS, full vs indexed baseline: time -15.7929%, peak requested storage -44.8232%, both 24/24 wins. Paired descriptive root-bootstrap mean differences: -2.8334 us, 95% interval [-4.05575,-1.79204]; -5167.67 B, interval [-7629.33,-3018.33]. Direct+packed fastest cut arm is descriptive (not a retrospective primary gate); full optimization lowest mean peak. Deferral alone increases latency on 24/24 roots despite saving storage on 19, ties 4, loss 1; extra coarse scans average 383.375. Direct eliminates zone map (mean cached zones 21.58 -> 0) but increases incidence unions from 337.42 to 756.5. Full mean allocations 85.29 ->18.125; max OOD peak requested 43720 ->24832 B. Packed hmax improves mean latency by 24.35% vs unmodified hmax and wins all 24.

Both cut and hmax policies certify all roots. All indexed variants have exact same expansion traces, 10.25 mean OOD expansions vs 15.04167 hmax (-31.85596%). Secondary gate FAIL: full cut remains 65.58% slower than packed hmax; timing wins 3/24. Mean peak requested bytes lower than packed hmax, but only 11 wins, 1 tie, 12 losses. No universal memory superiority. Do not compare absolute timing across V16/V17 cohorts.

## Audit

Independent set-based V15 reference audit for two reference arms x36 roots: 47,652 exact Bellman states, 2,795 bound certificates, 1,749 unavoidable cuts, 4,544 prefixes, 2,448 priority events, 432 snapshots, 13,392 fields, 2,346 proof steps; 28,616 residual distance coordinates. All pass. V17 native comparison (360 cases; repeated reference traces across factorial arms) matches 17,094 ordered events, 12,676 coarse bounds, 13,992 cut records, 2,160 snapshot proofs and 64,800 snapshot field comparisons. All intervals contain optimum. Direct-union and storage counts reconstructed from reference trace. Recomputed scans checked via distance-scan delta and nonnegativity, not independently recounted. ASan/UBSan full replay agrees on algorithmic fields and tracked allocation counts. Timing medians recomputed from raw samples; deterministic per-run non-timing result and event digest, full trace replay once per case. Tracking PMR peak requested bytes includes live container growth overlap, excludes allocator overhead, stack, shared graph bank and trace/output. Whole release process peak RSS 1792 KiB; no per-arm RSS or physical memory cap. ABI Node 48 B, Residual 64 B, Item 20 B, Engine 2368 B, Graph 460 B, Track 40 B, Result 1480 B. Timed region includes index init, allocation/destruction and trace digest, excludes input parsing and offline oracle/replay. Host-specific instrumented timings, no hardware/general speedup claim.

## Reproduce

Use bundled replay ZIP (prior/ contains V11 proof_planning.py, V12 anytime.py, V13 bounds.py, V15 indexed.py/experiment.py/audit.py); or copy those pinned files into prior/ next to source. Python with PyTorch import dependency, no model training. Assertions enabled.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python reference.py
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic native.cpp -o native
    ./native bank.txt 101 > native.txt
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer native.cpp -o native_san
    ASAN_OPTIONS=detect_leaks=1 ./native_san bank.txt 1 > sanitized.txt
    python compare.py

Git source blobs checked byte-identical to local: native.cpp ff9711045ac44bc5648edbb60e1979a06fdb8a2d; reference.py d1c6eccc69aee597f62bb690eeac09f2b4e05833; compare.py 2cda4d2b9876f4c7495ccffb9bb6c50ed9c5f8b3. Artifact SHA256: bank 3ff26ff63c21018e7f4833ee541776bc0b812a4ab5f3782666946c4c1b2b21a0; reference 3f19b298fb43f71de6bb5210e585583c1ac890b088cacea4b0370610f4e0e2dd; native f2f4113bef8190401281239dc9ce9908ecbdef4cd18cc4c2f509847cd5675659; sanitized 6e5c7482074d98dc6c578d42e2e9d6ce8810d524890698431ae72d3ae753e977; comparison 27d0ee1287ba023061c05d0642cbb58f489c03ba3b1af40b3893293bebf070db. The local package also includes deterministic generation script and baseline source. No deviations from main protocol except retaining only independent hmax/indexed reference arms as specified (not unnecessary scan reference).

Next: budget-aware certificate refinement and residual retention. A cheap hmax certificate can remain active until a stronger cut is worthwhile; allowing expansion before full refinement changes exact-order equivalence but not admissible pruning when complete OPEN is retained. Compare against packed hmax under fixed actual work/time and memory, with no pruning from learned priority. Evaluate whether recomputing evicted residuals from verified cut prefixes or storing compact initial-distance vectors saves actual allocation. Native BPD/MMS integration, Lean proof and external generalization remain outstanding.