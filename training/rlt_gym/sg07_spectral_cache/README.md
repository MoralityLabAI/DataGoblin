# SG-07: amortized exact certificate arithmetic

Optional CachedDualSession inherits the SG-05 verifier. Fixed task and (remaining depth, multiplier vector) determine a base and per-component integer increments; an active suffix mask adds the corresponding increments. An LRU caches those tables, another LRU caches fixed-multiplier scalar bounds. A separate overlay retains accepted proposal bounds, at most one entry per distinct proposed state and at most the proposal cap. No live frontier, dominance entry, or transcript is evicted. The resulting cache caps are counts, not a global memory bound.

Run from repository root with installed DataGoblin:
`python -m pytest training/rlt_gym/sg07_spectral_cache/test_spectral_cache.py`
`python training/rlt_gym/sg07_spectral_cache/sg07.py > sg07_summary.json`
`python training/rlt_gym/sg07_spectral_cache/sg07_ablation.py > sg07_ablation_summary.json`

The 60 independent generated development roots use fresh seeds, 36 d8/T6, 12 d9/T7, 12 d10/T8, equally across persistent, rotating, mixed. Three randomized arm orders and process CPU timings per root; summarize mean of root medians. All arms share exact heuristic incumbent, 2048 transitions, 24 proposals, 80 decisions. Always propose a reverse coordinate sweep from zero once per selected state, then expand. Lazy variant retains eight recent vectors globally and tries same-depth candidates; reuse iff exact improvement covers at least half the current state gap (rounded up, minimum one). A proposal must still be independently verified. This is a reuse heuristic, not a trained controller. Timing includes session init, bounds, selection and final receipt; excludes exact oracle, independent replay and separate tracemalloc runs. Nine predetermined memory roots. All 900 receipts replayed against original code; both cache-only arms match original decisions and receipts. Three targeted pytest tests passed (7680 candidate bounds, 7680 fixed bounds, 32 interleaved searches plus invalid inputs). Full repo suite and Ruff pending. No GPU, paid inference or native checkpoint.

| Arm | Mean CPU ms | Certified /60 | Mean gap | Mean transitions | Mean coordinate updates | Mean peak traced bytes |
|---|---:|---:|---:|---:|---:|---:|
| original |16.3258451|59|0.25|444.6333|38.75|51141.33|
| cache128 |3.94396993|59|0.25|444.6333|38.75|58776|
| cache12 |4.2170357|59|0.25|444.6333|38.75|45994.67|
| lazy128 |3.6989779|59|0.25|444.5|26.5|59175.11|
| lazy12 |4.00981007|59|0.25|444.5|26.5|47225.78|

All arms mean regret 0.20. Both lazy arms reused 137 vectors on 402 candidate checks, mean proposals 8.3333 versus 8.2833. Larger caches attained maximum 67 tables, 128 scalars; smaller attained 12 and 16 and evicted 1829 tables/12323 scalars (cache-only). Overlay maximum 24. Lazy128 wins versus original on 60/60 paired medians; cache12 wins 59/60. Single-host development timing; no generalization or mechanistic inference.

Post-hoc ablation on the same exposed 60 roots, three additional timing repetitions, 900 exact original receipt comparisons: original 15.5607 ms; linear recurrence with neither cache 17.3803; tables only 4.8671; scalars only 10.1594; both 3.6621. Caching drives the speedup on this workload. Cache-eviction counts do not bound total live search state or process RSS.

Raw receipt archive supplied separately; results JSON SHA256 550c1704a59a19caeae92b1659cfe4b7328d81849105dabe08ba7907a8d52644 and ablation JSON SHA256 dcccbcc6a0ddf2ce2f62a25c6645ea174f027d66d10bbba2b1574c7145e6f1d1. Executed code is preserved byte-identically; timings regenerated on another host will differ.
