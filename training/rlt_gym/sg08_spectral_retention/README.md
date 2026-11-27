# SG-08: learned certificate retention

Optional RetentionSession changes only recomputable SG-07 multiplier-table eviction; the original lightweight LRU remains unchanged and recommended. Scalar cache capacity 16, proof state and proposal overlay identical. Online mandatory-insertion policies: LRU; resident-frequency LFU; depth times resident frequency divided by age; fixed-first (evict nonfixed before fixed, LRU within class); learned linear rank; root-shuffled control. No unbounded history: resident counters, fixed keys (six prefix vectors per depth); request recording optional and unbounded when enabled. All slots hold one table, not equal byte counts. This is a finite offline synthetic development workload, no neural native-model performance claim.

Run from repo root in an installed environment with NumPy:
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg08_spectral_retention python -m pytest training/rlt_gym/sg08_spectral_retention/test_spectral_retention.py`
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache OPENBLAS_NUM_THREADS=1 python training/rlt_gym/sg08_spectral_retention/sg08.py`
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache python training/rlt_gym/sg08_spectral_retention/sg08_cap6.py`

Train 36 roots, 35,442 candidate labels from LRU eviction opportunities at 6/12/24 entries. Label depth times next 2C requests' hit count for that candidate. Ridge fit, 11 integer-quantized coefficients; shuffled targets within each root. Heldout exposed development 48 fresh roots: 24 d8/T6, 12 d9/T7, 12 d10/T8; equally divided persistent/rotating/mixed. Same request stream replayed at three capacities. Offline Belady with future next-use knowledge is optimum for misses under mandatory insertion; not a weighted-cost optimum. Weighted cost is sum of missed table depths times component count. Live CPU compares native LRU and six generic online policies at 12 slots; randomized order, three repeats, mean of root medians. Excludes offline fitting, exact oracle and replay. 1008 complete independent certificate replays, identical receipt and arithmetic work parity against simulator, 48/48 certified optimal. Tests also directly compare full request streams on 288 small runs and Belady against exhaustive cache-state DP on 2916 trace-cap pairs. Three targeted pytest tests passed; full repo and Ruff not run.

| Policy | Component steps C=6 | C=12 | C=24 | CPU ms C=12 |
|---|---:|---:|---:|---:|
| LRU |141733|74891|64096|4.7066|
| LFU |1738911|1622917|441645|13.0433|
| Weighted |1162029|473699|127582|7.5337|
| Fixed-first |96457|77157|64843|4.7479|
| Learned |113439|72997|65869|5.1880|
| Shuffled |1680896|1523593|968979|26.1932|
| Belady |88947|67621|62530|offline only|

Original native LRU 3.9379 ms. 66,775 requests. Separate post-hoc same-root six-entry timing (576 receipt comparisons): native LRU 3.9108 ms, generic LRU 4.6388, fixed-first 4.5559, learned 5.0242. Fixed-first reduces work 31.94% at six slots; learned reduces work 2.53% at twelve slots. Neither beats native LRU in mean CPU time at either measured capacity. No independent confirmation or RSS measurement. Raw timings and request streams provided separately. Main JSON SHA256 99bca7bff03e58809155d5b9c1206c985618887a1cda188d216f9baa95638b7b; cap6 JSON SHA256 271cc6ba354111e346a432e39edbede0a2c7c49726f1083645575d7c324a601c.
