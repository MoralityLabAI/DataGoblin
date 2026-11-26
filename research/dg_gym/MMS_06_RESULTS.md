# MMS-06 Results: Repaired Memory Accounting, Canonical Factor Maps & CPU Break-Even

Status: Public development only; local CPU hardware (`threads=2`), no paid compute, no sealed cohorts opened.
Pre-execution registration: `research/dg_gym/MMS_06_REGISTRATION.md` (commit `0fb4e54e760c0ae1b35f49bad83a28f87f97cd66`).
Audited baselines:
- `MoralityLabAI/DataGoblin`: `3fd7a38a90e3e7c0e977b29f45804bd4301ea032` (`research/dg-gym-v07-mem-fb-01-20261126`)
- `MoralityLabAI/LoopedTransformers`: `3dad5bd9f7b076fffdad9c717dbc98b04982cba7` (PR `#397`)

All historical MMS-01–05 receipts are preserved untouched; amended interpretations where the MMS-06 audit changes their scope are noted explicitly below.

---

## 1. Environment, Host Inventory, Clocks & Source Provenance

- **Host:** AMD Ryzen AI 9 365 (10C/20T), `47.12 GiB` physical RAM (`10.32 GiB` free at start), `97.55 GiB` commit limit (`9.72 GiB` free), `43%` baseline CPU load.
- **Software:** Python `3.12.14`, PyTorch `2.11.0+cu128` (CPU-only runs; GPU measurements unavailable), NumPy `2.5.3`, Windows 11 (`10.0.26200-SP0`).
- **Blocked / Unavailable:**
  - **Mathlib MMS-04:** remains **BLOCKED** locally (packed `mathlib-c3` corpus not staged on this machine).
  - **GPU throughput:** unavailable for CPU-only MMS-06 runs.
- **Empirical Clock Calibration (`mms06_lbf20m_summary.json`):**
  - `perf_counter` (`QueryPerformanceCounter()`): reported `1e-7 s`, observed min increment **`99.97 ns`**.
  - `process_time` (`GetProcessTimes()`): reported `1e-7 s`, **observed min and median tick increment = `15.625 ms` (`0.015625 s`)**. Timing blocks in LBF-20M were batched via `inner_reps` (`8/4/2`) across `5` randomized trials so block times exceed tick quantization by `5x–20x`.
- **Source SHA-256 Digests:**
  - `src/datagoblin/training_gym/mms_utility.py`: `973739e430500cf73007c62c6be13161272a118c6c6c27f12f053694a1fd4f2b`
  - `src/datagoblin/training_gym/factor_maps.py`: `7ad66bb7f464eddd1cff3e6cddfab8eb51584ca54ee1649c5f81a35c1cd19f07`
  - `src/datagoblin/training_gym/fast_gradient_bank.py`: `f675c652b76325ba34af37f9187309272e9b7d5de9956da900127f4151ded638`
  - `research/dg_gym/fit_bpd_map.py`: `3899e9577476e64cd421365f2e4f365b038f28cb213e312f1c456d5b6d8971a4`
  - `research/dg_gym/lbf20m_rebenchmark.py`: `0f86fb5410e40a292697d00c0f8500e1c34bbd3532e56243af4c83cd5f4afb6f`
  - `research/dg_gym/mms06_lbf20d_campaign.py`: `7d1a32d194e63d979a7559ea91a6a2515a3379aa66efbc0ce91205aab8c05633`
  - `research/dg_gym/aggregate_mms06.py`: `2fbdc85bdf8b17868b0a0660ab242000445265222258fa70f48531dd3033c7a9`

---

## 2. Code Repairs & Regression Evidence

1. **`UtilityDirectedScheduler` (`src/datagoblin/training_gym/mms_utility.py`, 6 regression tests in `tests/test_dg_gym_mms_utility.py`):**
   - Stores owned compact float32 copies (`arr.base is None`); verified against a view slicing a 100k-float base array and against post-insertion caller mutation.
   - Zero-byte `history_budget_bytes` rejects all insertions (no empty-record accumulation); explicit record-count + byte bounds enforced.
   - Covariance charges derived directly from `grad_mean.nbytes + grad_m2.nbytes` for configured `cov_dim`; receipt separates `owned_array_bytes`, `estimated_charge_bytes`, and `measured_process_rss_bytes`.
   - Added `ORDINARY_SITE = "__ordinary__"` with CPU/live-RAM feasibility filtering, budgeted `research_mode` exploration, abstention to ordinary training when exploitation lacks positive gain evidence, full `FeedbackContext` provenance + cross-basis invalidation, separate metric contexts per unit/scale, trajectory-bound action traces (verified negative feedback on site A stays independent of site B), and separated receipt fields (`empirical_gain`, `uncertainty_se`, `snr_multiplier`, `exploration_bonus_term`).
2. **Canonical Factor Maps (`src/datagoblin/training_gym/factor_maps.py`, `research/dg_gym/fit_bpd_map.py`, 3 regression tests in `tests/test_dg_gym_factor_maps.py`):**
   - **Code deduction confirmed:** `LoopedTransformers@3dad5bd9` (`research_gym/interp/mms.py:180-181`) normalized `U` without rescaling `V` by `n_j = ||U_j||_2`, causing equivalent factorizations `(V diag(a), diag(a)^-1 U)` to select different physical components.
   - Enforced `W_hat = (V U)^T`, `U'_j = U_j / n_j`, `V'[:,j] = V[:,j] * n_j`, explicit zero-component exclusion, sign canonicalization, and canonical `component_id` ordering. Verified exact selection and injected-gradient invariance under arbitrary nonzero rescalings (`1e-3..400`, sign flips) and permutations, plus portable manifest-relative paths (`runs/map.pt`), expanded provenance fields, and full preparation timing breakdown.
   - `fit_bpd_map.py` is accurately documented as sparse factorization of a checkpoint weight matrix via MSE + elementwise L1 (not behavioral BPD discovery).
3. **Vectorized `FastGradientBank` & Non-Overlapping `CostLedger` (`src/datagoblin/training_gym/fast_gradient_bank.py`, 3 regression tests):**
   - Replaced per-page Python loops (`for p in pages.unique().tolist():`) with single-pass vectorized BF16 flat arrays preserving exact collisions, `max_age` expiration, same-step skips, duplicate-key last-write-wins, hard byte caps, and resume parity.

---

## 3. LBF-20M Exact Caching Rebenchmark (Amended MMS-01–05 Scope)

**Direct measurement (12 fresh CSP roots, 5 randomized-order trials, exact answers & valid witnesses verified on every mode; process RSS `45.10 MB`, compiled base tables + global helper caches reset at every root/arm boundary):**

| Workload | `disabled_mdd_components` Q Wall (ms) | `zero_cap_instrumented` Q Wall (ms) | `cap_2KiB` Q Wall (ms) | `cap_4KiB` Q Wall (ms) | `cap_8KiB` Q Wall (ms) | `cap_12KiB` Q Wall (ms) | `cap_64KiB` Q Wall (ms) | `cap_12KiB` 95% Paired Root Wall CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cold_r1` (rep=1) | 77.88 (`+0.0%`) | 97.70 (`+25.4%`) | 72.06 (`-7.5%`) | 67.97 (`-12.7%`) | 68.56 (`-12.0%`) | **64.78 (`-16.8%`)** | **62.28 (`-20.0%`)** | `[-1.656, -0.561] ms/root` |
| `low_rep_r2` (rep=2) | 138.88 (`+0.0%`) | 172.51 (`+24.2%`) | 127.11 (`-8.5%`) | 124.90 (`-10.1%`) | 112.27 (`-19.2%`) | **110.50 (`-20.4%`)** | **110.99 (`-20.1%`)** | `[-3.586, -1.220] ms/root` |
| `high_rep_r4` (rep=4) | 281.30 (`+0.0%`) | 340.07 (`+20.9%`) | 249.90 (`-11.2%`) | 238.02 (`-15.4%`) | 225.22 (`-19.9%`) | **215.98 (`-23.2%`)** | **222.05 (`-21.1%`)** | `[-7.08, -3.57] ms/root` |

- **Eviction & working-set curve:** On `cold_r1`, peak query payload saturates at **`11,980 B`** (`11.7 KiB`); evictions drop monotonically from `175` (`2 KiB`) → `113` (`4 KiB`) → `24` (`8 KiB`) → `0` (`>= 12 KiB`), while separator messages drop `93.7%` (`1,422 → 89`) even on cold `repeat=1` due to intra-query terminal/component reuse. Setup wall time is `~210–246 ms` across 12 roots; verification is `~2.2–7.8 ms`.
- **Amendment to MMS-01–05 interpretation:** Our earlier `-27.3%` LBF-20M CPU savings figure compared bounded caching against `zero_cap_instrumented` (`mdd_query_cache` at `0 KiB`), which incurs a **`+20.9%` to `+25.4%` pure instrumentation/accounting tax** (`14.7–52.4 ms` recursive `payload_size` overhead) over truly disabled `mdd_components`. Against truly disabled `mdd_components`, bounded exact caching at `12 KiB` still delivers genuine, statistically robust savings (**`-16.8%` to `-23.2%` query wall time**, **`-15.6%` to `-21.5%` query CPU**, every 95% paired root bootstrap CI strictly negative).

---

## 4. `GradientBank` Profile & Vectorization Speedup

Direct profile on exact LBF-20D shape `(64,)`, `batch=128`, `4 MiB` (`capacity=29,127`, `4,194,288 B` owned), `240` steps (`30,720` queries, `19,402` exact hits, `0` collisions on BOTH implementations):

| Implementation | Total Wall (s) | Total CPU (s) | Read CPU (s) | Write CPU (s) | Speedup vs `3dad5bd9` |
| --- | ---: | ---: | ---: | ---: | --- |
| `LoopedTransformers@3dad5bd9` `GradientBank` | 11.3038 | 10.1563 | 4.0469 | 6.0469 | 1.0× |
| `DataGoblin` vectorized `FastGradientBank` | **0.1869** | **0.0781** | **0.0156** | **0.0625** | **130.0× CPU / 60.5× wall** |

Additional component profiles: canonical SVD refresh `4.69 ms` CPU/call; 60-step sparse MSE+L1 fit `~500 ms` CPU/call. Vectorizing `select_and_project_numpy` (using canonical `component_id` column order + stable `np.argsort`) further cut per-step feedback CPU by `~4–5×` (`~2.1s → ~0.48s` over 400 steps), lowering step CPU overhead ratios from **`15.9×` (MMS-01–05)** down to **`1.38×–1.64×`** for `dense`/`svd_frozen`/`svd_refreshed`!

---

## 5. Repaired LBF-20D Campaigns: Break-Even & Overfitting Confound

Cohorts strictly disjoint by isomorphism signature: `train_96` (`2c75dd5e...`), `feedback_dev_train_16` (`772d206b...`), `final_dev_16` (`9c0bd3c3...`). Paired seeds `(101, 107, 113)`, `threads=2`, `B_RAM = 4 MiB` bank (`4,194,288 B` owned; scheduler `<= 3.2 KiB` estimated charge; peak process RSS `~438–443 MB`).

### A. Pre-Overfitting Regime (`B_CPU = 1.5s`, vectorized projection — primary unbiased break-even test)

| Arm | Mean Upd | Mean Regret (`final_dev`) | $\Delta V_\pi$ vs Ord | Per-Seed $\Delta V_\pi$ (`101, 107, 113`) | Root SE (16 roots) | Step CPU Ratio | `C_tot` (s) | Wall (s) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `ordinary` | 169.7 | 1.0035 | +0.0000 | `0, 0, 0` | 0.0000 | 1.000× | 1.51 | 0.92 |
| `svd_frozen` | 119.0 | **0.9618** | **+0.0416** | `+0.1265, +0.0425, -0.0441` | 0.0552 | 1.429× | 1.51 | 0.94 |
| `dense` | 120.3 | 0.9782 | +0.0253 | `+0.1166, +0.0336, -0.0743` | 0.0462 | 1.411× | 1.51 | 0.84 |
| `retrospective_end_diagnostic` | 118.0 | 0.9856 | +0.0179 | `+0.1023, +0.0282, -0.0770` | 0.0462 | 1.443× | 1.51 | 0.83 |
| `svd_refreshed` | 123.7 | 0.9937 | +0.0098 | `-0.0052, +0.0677, -0.0331` | 0.0576 | 1.382× | 1.51 | 0.92 |
| `sparse_frozen` | 96.3 | 1.0068 | -0.0033 | `+0.0492, +0.0299, -0.0891` | 0.0629 | 1.780× | 1.52 | 0.85 |
| `sparse_refreshed` | 83.3 | 1.0266 | -0.0232 | `+0.0997, -0.0901, -0.0791` | 0.0728 | 2.059× | 1.51 | 0.89 |
| `shuffled_feedback` | 103.7 | 1.0406 | -0.0372 | `+0.0133, -0.0502, -0.0746` | 0.0635 | 1.649× | 1.51 | 0.83 |
| `utility_scheduler` | 111.3 | 1.0628 | -0.0593 | `-0.0102, -0.0109, -0.1568` | 0.0556 | 1.553× | 1.50 | 0.83 |

### B. Fixed Exposure (`updates = 400`, vectorized projection)

| Arm | Mean Regret (`final_dev`) | $\Delta V_\pi$ vs Ord | Root SE | Step CPU Ratio | `C_tot` (s) | `C_train / C_lookup / C_fb / C_map+ref / C_ver` (s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ordinary` | **1.0119** | +0.0000 | 0.0000 | 1.000× | 3.30 | `3.30 / 0.00 / 0.00 / 0.00 / 0.00` |
| `svd_frozen` | 1.0147 | -0.0029 | 0.0260 | 1.565× | 5.17 | `4.00 / 0.66 / 0.51 / 0.01 / 0.00` |
| `utility_scheduler` | 1.0189 | -0.0071 | 0.0140 | 1.770× | 5.84 | `3.92 / 0.59 / 0.51 / 0.05 / 0.56` |
| `shuffled_feedback` | 1.0216 | -0.0098 | 0.0240 | 1.735× | 5.73 | `3.92 / 0.53 / 0.48 / 0.08 / 0.51` |
| `svd_refreshed` | 1.0220 | -0.0102 | 0.0331 | 1.644× | 5.43 | `4.13 / 0.62 / 0.58 / 0.10 / 0.00` |
| `sparse_refreshed` | 1.0276 | -0.0157 | 0.0262 | 1.959× | 6.47 | `3.96 / 0.53 / 0.52 / 1.46 / 0.00` |
| `dense` | 1.0326 | -0.0207 | 0.0150 | 1.421× | 4.69 | `3.77 / 0.73 / 0.20 / 0.00 / 0.00` |
| `sparse_frozen` | 1.0343 | -0.0225 | 0.0269 | 1.555× | 5.14 | `3.79 / 0.60 / 0.54 / 0.20 / 0.00` |
| `retrospective_end_diagnostic` | 1.0485 | -0.0366 | 0.0267 | 1.538× | 5.08 | `3.83 / 0.70 / 0.54 / 0.00 / 0.00` |

### C. Overfitting Regime (`B_CPU = 4.0s`, pre-vectorized projection — preserved diagnostic showing early-stopping confound!)

At `B_CPU = 4.0s`, `ordinary` executed **`964` updates** and **overfit** on the 96 training roots (`mean_regret = 1.1256`, worse than its own `1.0119` at `400` updates!), causing slower arms (`dense`: `609` upd, `1.0067`, `+0.1189` on 3/3 seeds; `sparse_frozen`: `311` upd, `0.9705`, `+0.1551` on 3/3 seeds) to beat `ordinary` **via implicit early stopping / fewer updates rather than superior learning efficiency**. This confound is explicitly separated from the pre-overfitting `B_CPU = 1.5s` break-even test.

---

## 6. Direct Measurements vs Deductions vs Unresolved Hypotheses & Next Experiment

- **Direct Measurements:**
  1. `FastGradientBank` eliminates `99.2%` of bank lookup/write CPU (`130.0×` speedup vs `3dad5bd9`), cutting small-model step overhead from `15.9×` to `1.38×–1.64×`.
  2. Exact caching on LBF-20M (`12 KiB`) delivers genuine `-16.8%` to `-23.2%` wall savings against truly disabled `mdd_components` across cold and repeated workloads.
  3. On pre-overfitting LBF-20D (`B_CPU = 1.5s`), `svd_frozen` (`+0.0416`, 2/3 seeds) and `dense` (`+0.0253`, 2/3 seeds) show noisy positive mean $\Delta V_\pi$ (within 1 root-SE) and **fail the registered 3/3-seed promotion gate**; at fixed exposure (`400` updates), `ordinary` (`1.0119`) slightly leads all memory arms (`1.0147–1.0485`); `utility_scheduler` (`-0.0593` at `1.5s`) is dragged down by `~11%` verifier overhead (`C_verifier = 0.16s`).
  4. In the 2×2 method-by-refresh grid, **learned sparse MSE+L1 factors do not outperform canonical SVD** on either fixed exposure or fixed CPU, and periodic sparse refitting (`C_refresh = 1.46s`) severely hurts fixed-CPU value.
- **Unresolved Hypothesis & Next Justified Experiment:**
  - Because `FastGradientBank` lookup/feedback is now only `~0.4–0.5 ms/step` while a `hidden=64` LBF-20D step is `~8 ms`, the true **overhead break-even region (`< 1.05×` step overhead)** lies on larger recurrent models (`>= 100 ms/step`, e.g., `RMP` `LoopedDecoderLM` `d=128` or `Mathlib TRM-XS` once `mathlib-c3` is staged) where verifier calls are amortized over larger windows (`>= 200` steps) and training does not overfit within `400` updates. **Do not promote MMS gradient replay yet**; next test `FastGradientBank` + `canonical_svd_factors` on `RMP pointer_chase` (or staged `mathlib-c3`) with `>= 8` seeds and early-stopping-protected ordinary controls.

## 7. Replay Commands & Receipts

- Receipts:
  - `research/dg_gym/receipts/mms06_lbf20m_summary.json`
  - `research/dg_gym/receipts/mms06_lbf20d_vec1p5_summary.json`
  - `research/dg_gym/receipts/mms06_lbf20d_4s_summary.json`
- Replay:
  ```powershell
  PYTHONPATH=src python research/dg_gym/lbf20m_rebenchmark.py --lt-repo <LoopedTransformers@3dad5bd9> --roots 12 --trials 5 --inner-reps 8 --out runs/mms06_lbf20m
  PYTHONPATH=src python research/dg_gym/mms06_lbf20d_campaign.py --lt-repo <LoopedTransformers@3dad5bd9> --seeds 101 107 113 --threads 2 --cpu-budget 1.5 --fixed-updates 400 --bank-mib 4.0 --out runs/mms06_lbf20d
  PYTHONPATH=src python research/dg_gym/aggregate_mms06.py --lbf20m research/dg_gym/receipts/mms06_lbf20m_summary.json --lbf20d-4s research/dg_gym/receipts/mms06_lbf20d_4s_summary.json --lbf20d-vec research/dg_gym/receipts/mms06_lbf20d_vec1p5_summary.json
  ```
