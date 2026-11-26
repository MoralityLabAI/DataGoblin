# MMS-06 Pre-Execution Registration: Memory Accounting, Canonical Factor Maps & CPU Break-Even

Status: Public development only; local CPU hardware, no paid compute.
No sealed cohorts are opened. All historical MMS-01–05 receipts are preserved untouched.

## 1. Provenance & Resource Plan

- **DataGoblin Base Commit:** `3fd7a38a90e3e7c0e977b29f45804bd4301ea032` (`research/dg-gym-v07-mem-fb-01-20261126`).
- **Audited LoopedTransformers Baseline:** `3dad5bd9f7b076fffdad9c717dbc98b04982cba7` (PR `#397`).
- **Refreshed Host Inventory (2026-11-26 21:02 GMT+8):**
  - CPU: AMD Ryzen AI 9 365 (10 physical / 20 logical cores), 43% baseline load.
  - Physical RAM: `47.12 GiB` total, `10.32 GiB` free.
  - Commit Charge: `97.55 GiB` limit, `9.72 GiB` free.
  - Runtime: Python `3.12.14`, PyTorch `2.11.0+cu128`, NumPy `2.5.3`, Windows 11.
  - Clocks: `perf_counter` (`QueryPerformanceCounter`, reported res `1e-7 s`), `process_time` (`GetProcessTimes`, reported res `1e-7 s`, OS tick quantum measured empirically in every benchmark).
- **Execution Constraints:**
  - Constant `--threads 2` for all CPU training comparisons; sequential execution of arms to prevent memory-bandwidth contention.
  - Hard total `scheduler + bank` RAM cap enforced by runner (`B_RAM <= 16 MiB` for small-model LBF-20D sweeps; total live process RSS measured via `psutil` including child processes).
  - Mathlib MMS-04 remains **BLOCKED** locally (packed `mathlib-c3` corpus not staged).
  - GPU measurements are **unavailable** for CPU-only runs.

## 2. Registered Primary Estimand & Complete Cost Ledger

For any policy $\pi$ under fixed total process CPU budget $B_{\mathrm{CPU}}$ and hard live memory cap $B_{\mathrm{RAM}}$, starting from identical initial weights, optimizer state, and RNG stream:

$$\Delta V_\pi(B_{\mathrm{CPU}}, B_{\mathrm{RAM}}) = V(\theta_\pi(B_{\mathrm{CPU}}, B_{\mathrm{RAM}})) - V(\theta_{\mathrm{ordinary}}(B_{\mathrm{CPU}}, B_{\mathrm{RAM}}))$$

where on LBF-20D $V(\theta) = -\mathrm{mean\_regret}(\theta; \mathcal{D}_{\mathrm{final\_dev}})$ (higher is better; positive $\Delta V_\pi$ means lower held-out Bellman regret than ordinary training at the same CPU budget).

Every run records the non-overlapping process-CPU ledger (plus separate wall-time columns):

$$C_{\mathrm{total}} = C_{\mathrm{train}} + C_{\mathrm{map}} + C_{\mathrm{lookup}} + C_{\mathrm{refresh}} + C_{\mathrm{feedback}} + C_{\mathrm{scheduler}} + C_{\mathrm{verifier}} + C_{\mathrm{other}}$$

Timer scopes are mutually exclusive (hook/lookup/refresh/scheduler/feedback timers pause the enclosing step timer or are subtracted without double counting). Every arm uses the exact same `C_total >= B_CPU` stopping check before launching the next step, reporting unused budget or overshoot explicitly. Fixed-exposure comparisons (`updates = 400`) are run and reported separately.

## 3. LBF-20D Cohort Separation, Arms & Independence Unit

- **Cohorts (from `lbf20d_learning.tasks()`, isomorphism-signature disjoint):**
  - `train`: `cohorts['train'][:96]` (96 roots, 128 sampled states/root = 12,288 rows).
  - `feedback_dev_train`: `cohorts['dev6'][:16]` (16 roots, used exclusively for scheduler feedback; never evaluated as final outcome).
  - `final_dev`: `cohorts['dev6'][16:32]` (16 roots, strictly unsteered held-out development evaluation; never read by any scheduler, map fitter, or bank).
- **Paired Seeds:** `(101, 107, 113)`.
- **Registered Arms (all at `threads=2`, `rank=16`, `topk=4`, `ratio=0.2`, `B_RAM = 4 MiB` bank + `256 KiB` scheduler cap, `B_CPU = 6.0 s` fixed-CPU and `400` updates fixed-exposure):**
  1. `ordinary`: zero-memory baseline (`mms_mib=0`, no bank, no map).
  2. `dense`: vectorized `FastGradientBank`, dense historical gradient replay (`mode='dense'`).
  3. `svd_frozen`: canonicalized SVD computed once at shared starting checkpoint $\theta_{\mathrm{start}}$, frozen.
  4. `svd_refreshed`: canonicalized SVD refreshed every `64` steps (charged to `C_refresh`).
  5. `sparse_frozen`: learned sparse factorization (MSE + elementwise L1 via `fit_bpd_map.py`) fit once at $\theta_{\mathrm{start}}$, frozen (fit charged to `C_map`).
  6. `sparse_refreshed`: learned sparse factorization refreshed every `64` steps (every fit charged to `C_refresh` + `C_map`).
  7. `retrospective_end_diagnostic`: separately labeled non-causal diagnostic using sparse factors fit on end-of-training ordinary checkpoint (excluded from promotion).
  8. `utility_scheduler`: repaired `UtilityDirectedScheduler` choosing among `(__ordinary__, block@0.5MiB, block@4.0MiB)` using `feedback_dev_train` every `40` steps (feedback evaluation charged to `C_feedback` + `C_verifier`; scheduler work charged to `C_scheduler`).
  9. `shuffled_feedback`: identical scheduler and cost as `utility_scheduler`, with feedback deltas randomly permuted across actions.
- **Unit of Independence:** The 16 `final_dev` roots are structurally disjoint CSP/DNF instances; paired per-root regret differences are reported alongside seed-level paired means across the 3 seeds (explicitly noting shared training roots across seeds; 3 seeds are descriptive development evidence).
- **Promotion Rule:** The `1.2x` per-step CPU overhead ratio is an engineering diagnostic only. Promotion requires $\Delta V_\pi(B_{\mathrm{CPU}}, B_{\mathrm{RAM}}) > 0$ on mean and on all 3 paired seeds over `ordinary`, and beating `shuffled_feedback` for scheduler arms.

## 4. LBF-20M Exact Separator-Query Cache Rebenchmark Protocol

- **Modes:**
  - `disabled`: native `Factorized(..., mode='mdd_components')` (zero cache lookup or insertion overhead).
  - `zero_cap_instrumented`: `mode='mdd_query_cache'` with `CachePool(0)` (retains instrumentation tax control).
  - `bounded_exact`: `mode='mdd_query_cache'` with `CachePool` at `0, 2, 4, 8, 12, 16, 32, 64 KiB`.
- **Workloads & Roots:** `12` fresh CSP roots (`lbf20m_factor.fresh()[:12]`), tested under `cold` (`repeat=1`), `low_rep` (`repeat=2`), and `high_rep` (`repeat=4`) query streams (verifying evictions occur at `2–12 KiB`).
- **Timing & Statistical Hygiene:**
  - `5` independent randomized-order trials per workload with explicit warmup and full cache/helper resets (`lift.cache_clear()`, `project.cache_clear()`) at every trial/arm boundary.
  - Inner batch repetitions scale timing blocks to minimize Windows `GetProcessTimes` quantization; observed minimum nonzero clock tick increments are logged.
  - Separate timing columns for `solver_setup`, `query_execution`, `cache_accounting`, `verification`, and `complete_pipeline`.
  - Paired uncertainty computed over the `12` independent roots (respecting root dependence across trials).
- **Verification:** Exact support answers and valid witnesses verified on every mode and root.

## 5. Planned Code Repairs & Regressions

1. `UtilityDirectedScheduler` (`src/datagoblin/training_gym/mms_utility.py`): owned compact coordinate copies, zero-byte empty-record prevention, actual covariance byte accounting, strict validation, ordinary-training action + budget/headroom feasibility filter, full `FeedbackContext` provenance + cross-basis invalidation, trajectory-bound action traces, separate metric contexts, separated gain/exploration/SNR/SE receipt fields.
2. Canonical factor maps (`src/datagoblin/training_gym/factor_maps.py`, `research/dg_gym/fit_bpd_map.py`): exact $U'_j = U_j / \|U_j\|_2$, $V'_{:,j} = V_{:,j} \|U_j\|_2$ canonicalization, explicit zero-component exclusion, stable physical tie-breaking, reciprocal-rescaling & permutation invariance tests, portable manifest-relative paths, expanded provenance manifests, full preparation timing breakdown.
3. Vectorized `FastGradientBank` (`src/datagoblin/training_gym/fast_gradient_bank.py`): profiled against `3dad5bd9` `GradientBank`, vectorized flat-buffer lookup/write eliminating per-page Python loops, strict parity tests for collisions, expiration, eviction, duplicate keys, resume, and observe-parity.
