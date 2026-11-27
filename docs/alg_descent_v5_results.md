# ALG-DESCENT-05 native packed cache (2026-11-26)

Pre-run protocol: [26386098](https://github.com/MoralityLabAI/DataGoblin/commit/26386098c064e0f90af9e0ae76e672d7290a2d41). Synthetic public-witness polynomial certificate routing; no factor discovery, DLP solving or cryptographic security claim. C++17 selected because Rust/Cargo unavailable. Debian g++ 14.2.0, AMD EPYC 9V74 host, Python 3.13.5. Release -O3. 24 fresh IID roots (150000..150023), 32 fresh OOD roots (160000..160031). v4 train-only prior and public symbolic scores, unchanged generator and eligibility checks. Candidate ordering precomputed; native code does not run a validity oracle. Exact native verification charges all attempted coefficient operations; cache hit costs one. Budget 3*n^2 work, 8*n calls.

## Reproduce

From repo root (Python with v4 dependencies, Linux C++17 compiler):

    SRC=src/datagoblin/research/alg_descent_v5
    mkdir -p /tmp/alg-descent-05
    g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic "$SRC/native.cpp" -o /tmp/alg-descent-05/native
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/export_bank.py" /tmp/alg-descent-05 36
    /tmp/alg-descent-05/native /tmp/alg-descent-05/bank.bin 41 > /tmp/alg-descent-05/native.csv
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$SRC/native.cpp" -o /tmp/alg-descent-05/native_san
    ASAN_OPTIONS=detect_leaks=1 /tmp/alg-descent-05/native_san /tmp/alg-descent-05/bank.bin 1 > /tmp/alg-descent-05/sanitized.csv
    python "$SRC/check_receipt.py" /tmp/alg-descent-05

The slot size parameter is ABI-specific (observed 36 on this host). Native program reports actual sizeof Slot. Re-run Python reference with reported size if different. All three source files were read back from GitHub and matched local git blob hashes:
- native.cpp 4bde1c7633b08e50a1ceda1b735e6928f68dc595
- export_bank.py 6fa5405bd43ebf869c2c621283b48064752657bb
- check_receipt.py 8fd78422d307f666f1db6c1e6413756b97f14c84

Local artifacts SHA256:
- bank.bin c44aec4ab11c86f6e43ae311b4c139c215b5a2a83d6527a9ef028d97a2a41101
- native.csv 5f20bede806c3959c322472ac5772608b7fdb549318206d094409e10bc13bc6d
- reference.json 379423d7312049578ef5520187c4a4d7e567901646fb23e72859219938cc98d5
- sanitized.csv 4c2c18698426428c184919eb615bbc259f7a48d1147dc0acd75b5f6b2c41713d
- audit.json 5f36a2e81f9999795e08f6d9c21748683459e7abb41a87c811785081cfad11cb

Timing-containing artifacts have run-specific hashes. Raw binary bank and receipts available in local package; regenerate with source.

## Native design and correctness

Exact 4-bit coefficient keys, 25 packed bytes + length; max degree 48, p in {2,3,5,7,11}. Cache scoped per root and prime, equality compares full canonical packed key. 36-byte fixed entry (key, padding, recency stamp, observed saved work), preallocated fixed count floor(cap/36); no dynamic cache allocation after construction. 32-byte cache control, 104-byte fixed pending stack (48 uint16 IDs and two uint32 counters). Both policies track same pending stack for positive caps. Pending score = pending multiplicity * (observed subtree work - 1); select minimum value with LRU tie break, refuse insertion if its value is smaller. Linear scans of slots and pending stack. Store trusted in-process closure receipt only after exact full successful subtree. Candidate IDs strictly earlier with lower degree, accepted child polynomials checked against indexed node key. Cache equality is exact, not a fingerprint. Root bank validity/order comes from trusted generator; parser checks canonical lengths and residues; not designed as a hardened arbitrary hostile-bank service.

616 root-policy combinations, 41 randomized-order timing repetitions each = 25,256 timed executions; all closed. All 4,312 fields (closure, work, calls, rejects, hits, occupancy peak, pending peak) agree with independently replayed Python reference using coefficient convolution and OrderedDict. 5,880 packed-key mutation checks, malformed degree/residue/corruption fixtures, zero/one-slot controls. ASan/UBSan full replay passes, all 616 receipts equal excluding timings. Budget violations zero. Total native receipt rejections 456 across arms. This verifies the native implementation against the independent reference for the generated banks, not a machine-checked proof.

## Results

OOD root-average work and mean of per-root median warmed latency (microseconds):

| Entry cap | Slots | Allocated | LRU work | Pending work | LRU us | Pending us |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1029.375 | - | 3.039 | - |
| 64 | 1 | 36 | 885.438 | 885.438 | 2.583 | 2.607 |
| 128 | 3 | 108 | 849.062 | 828.188 | 2.394 | 2.369 |
| 256 | 7 | 252 | 803.188 | 799.719 | 2.199 | 2.243 |
| 512 | 14 | 504 | 797.156 | 797.156 | 2.207 | 2.211 |
| 1024 | 28 | 1008 | 797.156 | 797.156 | 2.354 | 2.354 |

All OOD arms 32/32 closed. At 128 B pending wins/ties/losses 16/16/0; mean difference -20.875, descriptive paired root-bootstrap 95% [-33.15625,-10.8125]. At 256 B 3/29/0, -3.46875, interval [-8.625,0]. Other caps all tie work. At 128 B pending mean latency difference -24.84 ns but paired median difference +14.5 ns (15 faster,17 slower roots); at 256 B +44.22 ns mean and +30 ns paired median. No two-capacity latency gate pass. Differences are small, timing is host-specific and not statistically conclusive. 256 B LRU saves ~22.0% coefficient work and ~27.6% mean per-root median time vs native no-cache. 512 B yields slightly less work and slightly higher mean latency; 1024 B no further work improvement. IID mean work no-cache 221.125; 128 B LRU 188.75, pending 185.083; all other capacities tie between policies.

## Actual memory and timing boundaries

Entry allocation excludes allocator metadata; add 136 bytes of fixed cache control + pending stack (e.g. 244 B at 128 B cap; 388 B at 256 B). The entry cap is not a process physical cap. Whole native max RSS 24,064 KiB (~23.5 MiB), includes shared loaded banks and process/runtime/benchmark allocations. No isolated per-policy private memory or PSS measure. Not comparable directly to v4 fork-worker Python RSS. No dynamic allocations by cache and pending stack after construction; native bank and measurement vectors allocate separately; recursion frames excluded. Max observed OOD pending entries 5. Timing includes creation/destruction of cache and exact routing/verification; randomized arms within root for each repetition. Shared bank 4,811,263 bytes. Train-only prior construction 0.446 s; evaluation bank construction/ranking 3.628 s; export 0.227 s; independent reference 0.114 s; native bank load 0.077 s. These common preparation phases dominate a single run; no end-to-end native generator speedup. First trial timing is not a process-cold benchmark.

## Decision

PASS exact native replay and native LRU work/time improvement on this host. FAIL registered pending-aware combined gate despite conditional work improvement. No native BPD/MMS or neural training. Next: intern canonical polynomial keys at bank load using exact comparisons, carry compact IDs into pending frontier and cache (e.g. 12-byte ID/stamp/saved entry), track pending multiplicities with bounded arrays or compressed counters, compare at matched total memory including intern table. Profile admission scans and remove dominated costs; benchmark bank construction/ranking too. For BPD, test incremental conditional validity prediction on genuinely held-out corruption families before routing; use symbolic prior as control and charge feature/score time. No claims about external generalization.
