# SG-10: quotient and vectorized certified expansion

Optional BatchedBankSession uses the SG-09 split bank. For a fixed suffix S, all actions A producing the same union S' incur the same last-use cost. Keep the highest-utility action per union, preserving the earliest enumerated action on ties. Skip already dominated groups before evaluating their bounds. Terminal expansion keeps the earliest maximizing action. Charge all original actions against the same transition cap. Original proposal overlay, proof state and event format remain intact. NumPy is used for unions and fixed-bank lookup batches; grouping, utilities after grouping, pruning, arbitrary multipliers and certificate state remain Python. Batch cache accesses take a snapshot of scalar hits before inserting misses, so LRU ordering may differ. Values do not. The hybrid uses NumPy for at least 64 actions or at least 16 missing bounds, respectively. The threshold was chosen before the run. Signed integer bank and vector operations apply to the existing validated d,T <= 10 task domain; arbitrary proposals retain Python integer evaluation. Original default unchanged.

Run from installed repository:
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg10_spectral_batch OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest training/rlt_gym/sg10_spectral_batch/test_spectral_batch.py`
`PYTHONPATH=training/rlt_gym/sg07_spectral_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg10_spectral_batch/sg10.py`

60 new exposed synthetic development roots, 20 per regime; 30 d8/T6, 15 d9/T7, 15 d10/T8. Same lazy policy, 12 dynamic table slots, 16 scalar slots, split bank, initial incumbent and caps 2048 transitions/24 proposals/80 decisions. Three randomized timing orders per root. CPU is mean of root medians, includes bank setup, policy, search and receipt, excludes external oracle and independent replay. Nine separate tracemalloc roots, not RSS. Five targeted tests passed (13,479 quotient comparisons, 46,080 repeated scalar/batch comparisons including synthetic overlay lookup fixtures, 36 interleaved searches checking live and dominance state, ties and cap, 4,096 extreme-range bounds). The synthetic overlay fixture tests lookup precedence, not proposal validity. All 720 timed full receipts matched and replayed through SG-05. All 60 certified optimal, zero mean gap and regret. 47 required expansions. Full repository pytest and Ruff unavailable; Rust unimplemented.

| Arm | CPU ms | Wins vs scalar /60 | Expanding roots CPU ms | Mean peak traced bytes | Bank queries |
|---|---:|---:|---:|---:|---:|
| scalar |2.784249083|0|3.243643426|57742.67|88965|
| dedup |2.585151367|49|2.993566809|61636.44|70456|
| batch |2.506502433|51|2.891035787|82920|69952|
| hybrid |2.509827067|48|2.893985511|82294.22|69952|

First repetition counts: 25,716 charged actions, 2,629 terminal. 23,087 nonterminal actions collapse to 11,539 unique union groups; 37 already dominated, 11,502 candidate bounds including cached hits. Batch 277 vectorized batches/11,385 missing masks; hybrid 237/10,937. Bank queries count multiplier-vector/mask evaluations, not total operations. All arms 12,615 dynamic steps plus 21,510 bank steps. Batch saves 9.98% CPU vs scalar and 3.04% vs dedup, 21.37% fewer bank queries, with higher memory. Same entry capacities, not matched bytes. Single host, no native-model or general performance claim.

Source blob hashes verified against executed files: spectral_batch cbbe0d1d68860864b5bc49be04591764db14737d, sg10 ff6a3f414589af1555b2743ecb3e7be5f647e10a, test 0259fb5673daf9626a933a7a3c0c3461f71dff5c. Raw results JSON SHA256 f914b9e118c89c9c5923c85f3e23ca5cd33c97e899c40f32fd90d714736b6b4c. Archive sg10_cpu_campaign.zip SHA256 21755e8f2b18b2fe692453ff695744bf1f3065826a2c139f51635b7f139f3bde provided with session; generated timing runs not committed. No GPU or paid inference.
