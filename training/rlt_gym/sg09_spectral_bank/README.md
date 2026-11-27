# SG-09: prefix-shared spectral table bank (development)

The fixed multiplier vectors share depth prefixes. For each vector, maintain running component contributions and a constant, and append each new prefix's base and deltas to a contiguous signed integer bank. Identical prefix keys are deduplicated. The bank's recurrence cost is O(M T d), with M=6 here. The optional split representation precomputes subset sums for two halves of the mask: O(M T (2^floor(d/2)+2^ceil(d/2))) storage and preprocessing, then two lookups and an addition per vector. The ordinary bank uses O(M T d) storage and O(popcount(mask)) evaluation. Dynamic proposals still use bounded SG-07 arithmetic caching, with exact Python integers; accepted-proposal overlay, scalar cache, state and receipt remain unchanged. The bank is an optional session type, constructed only with the validated task and internally generated fixed multipliers. It does not reinterpret arbitrary student multipliers as int64.

Run from repo root in installed environment:
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg09_spectral_bank python -m pytest training/rlt_gym/sg09_spectral_bank/test_spectral_bank.py`
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg09_spectral_bank/sg09.py`

60 fresh exposed development roots, equal across persistent/rotating/mixed, 30 d8/T6, 15 d9/T7, 15 d10/T8. Same SG-07 lazy policy and budgets; bank arms retain 12 dynamic table slots and 16 scalar slots in addition to fixed bank. LRU128 uses 128 slots of each. Three randomized arm orders per root; process CPU mean of root medians includes construction and receipt, excludes external oracle/replay and separate memory/microbenchmark. All 720 timed full receipts identical and independently replayed. All optimum intervals checked against exact oracle. 58/60 certified, mean gap 0.5, regret 0.216667. Three targeted tests passed: 38,400 exact fixed-vector comparisons, exhaustive fixed-bank checks at small dimensions and interleaved proposal replay, subset sums and integer range fixture. Full repository suite and Ruff not run.

| Arm | CPU ms | Dynamic steps | Bank steps | Mean traced peak bytes (9 roots) |
|---|---:|---:|---:|---:|
| LRU12 |3.851877833|99667|0|32576.44|
| LRU128 |3.547936917|79807|0|45557.78|
| Prefix |3.177035650|13949|21510|41535.11|
| Split |2.898993367|13949|21510|56267.56|

Prefix wins 50/60 paired medians versus LRU12, split 48/60. Split saves 24.74% mean CPU vs LRU12. Both bank arms reduce counted component-depth arithmetic from 99,667 to 35,459 across first runs (64.42%); subset-sum construction/additions, allocation and metadata not included in that work proxy. Mean bank payload 3258.93 bytes prefix, 15093.33 bytes split, maxima 4312 and 25088. These figures omit Python metadata; separate tracemalloc includes Python-managed allocations but not total RSS. Capacities are not equal byte budgets. Single-host development timing, not a general guarantee.

Nine full-depth mask batches (256/512/1024 masks), exact equality of NumPy and scalar result. Five repeats; mean root median scalar 0.685204 ms, NumPy 0.019089 ms. Excludes bank construction, precreated NumPy view/offsets/masks; scalar path converts masks to list and checks input. NumPy produces an array, scalar a list. This is a microbenchmark, not a search-level vectorization result; NumPy is not used in timed search. Rust compiler unavailable; no Rust implementation/benchmark claimed. Original LRU remains unchanged. Raw archive provided separately; generated runs not committed. Results JSON SHA256 102ed2e9fdc6e058e3280fb66531fc9d28f57915c1208773b160629dba6eff31. Archive SHA256 586f6dddbe20c480080e7bfc260bb30feb86ab9295b7b59d4231ad7a90c7eef4.
