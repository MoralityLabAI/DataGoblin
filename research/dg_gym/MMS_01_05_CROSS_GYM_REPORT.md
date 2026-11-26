# Cross-Gym Mechanistic Memory Scaling Report (MMS-01–05)

Status: Public development only; local CPU hardware, no paid compute.
No sealed cohorts opened; all historical receipts and negative results preserved.

## 1. Repository Integration & CI Audit

All five target PRs were fetched at their latest remote heads and audited:

| Repository | PR | Head Branch | Commit | Base / Stacking | CI & Local Verification |
| --- | ---: | --- | --- | --- | --- |
| `MoralityLabAI/LoopedTransformers` | #397 | `research/mms-bpd-existing-gyms-20261126` | `3dad5bd9` | `main` | 12/12 remote CI green; **17/17 local MMS unit tests pass** (`test_mms.py`, `test_lbf_mms.py`, `test_tactic_mms.py`) |
| `MoralityLabAI/HybridTRMLDT` | #31 | `research/mms-router-storyworld-20261126` | `4f291b55` | `master` | 10/10 remote CI green (`CodeQL`, `smoke`, `matched-heldout`, `fixture-and-protocol`) |
| `MoralityLabAI/HybridTRMLDT` | #32 | `research/mms-mcp-spatial-20261126` | `7151fc38` | `gym/mcp09-prospective-planning` | Remote CI green (`contracts`, `cpu-smoke`, `smoke`); stacked dependency respected |
| `MoralityLabAI/BPD` | #157 | `research/mms-native-rp03-20261126` | `58d00a7a` | `research/bpd-rp03-stable-steering-20261126` | Mergeable (`clean`); stacked dependency respected |
| `MoralityLabAI/DataGoblin` | #80 | `research/dg-mms-mod7-20261126` | `a97e4af0` | `master` | **3/3 local MMS unit tests + `ruff check` pass** |

## 2. Host Resource Inventory & Explicit Limits

Measured before launching sweeps:
- **Host:** AMD Ryzen AI 9 365 (10C/20T), `47.12 GiB` physical RAM, `94.24 GiB` commit limit.
- **Available Headroom at Launch:** **`5.56 GiB` free physical RAM** (`41.56 GiB` in use), **`9.71 GiB` free commit** (`84.53 GiB` committed), **`100%` CPU load** from concurrent host processes (including an external Python process holding `2.15 GiB` WS).
- **Resource Discipline:** Because free physical RAM was `5.56 GiB`, `4–8 GiB` bank allocations were **not** launched (preventing host OOM/thrashing). All sweeps ran sequentially at constant `--threads 2` within `< 1 GiB` process RSS.
- **Local Data Limit (Mathlib MMS-04):** `lcgrt.tactic_lm` and `lcgrt.tactic_mms` (native `loop`/`hrm`/`trm` clock observe parity, ACT carry resume, and repetition pool) passed all 5 local unit tests in `LoopedTransformers#397`, but the packed `mathlib-c3` corpus is not staged on this machine, blocking live Mathlib verifier runs locally.

## 3. Empirical Results Across Gyms

### A. LBF-20D (Bellman Regret, `updates=240` fixed exposure vs fixed CPU, `threads=2`, `seed=101`)

| Arm | `mms_mib` | Mean Regret | Expected Queries | Train CPU (s) | Wall (s) | Map Kind | Map Cos | Hits / Queries | Admitted | Alloc (B) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `zero_0m` | 0 | 1.2251 | 3.8762 | 1.438 | 0.792 | none | — | 0 / 0 | 0 | 0 |
| `obs_4m` | 4 | 1.2251 | 3.8762 | 24.453 | 12.707 | `svd_seed_proxy` | 0.7849 | 19,376 / 30,720 | 0 | 4,191,984 |
| `bpd_svd_4m` | 4 | **1.1688** | **3.8199** | 22.891 | 13.636 | `svd_seed_proxy` | 0.7863 | 19,376 / 30,720 | 18,881 | 4,191,984 |
| `bpd_svd_16m` | 16 | **1.1688** | **3.8199** | 21.250 | 10.868 | `svd_seed_proxy` | 0.7863 | 19,376 / 30,720 | 18,881 | 14,970,816 |
| `bpd_learned_16m` | 16 | 1.2444 | 3.8955 | 19.609 | 10.138 | `native_export` | 0.7795 | 19,376 / 30,720 | 18,891 | 14,970,816 |
| `dense_16m` | 16 | 1.1754 | 3.8265 | 22.016 | 11.316 | `svd_seed_proxy` | 0.7849 | 19,376 / 30,720 | 18,857 | 14,970,816 |
| `sham_16m` | 16 | 1.2217 | 3.8728 | 20.469 | 11.359 | `svd_seed_proxy` | 0.7848 | 19,376 / 30,720 | 18,876 | 14,970,816 |
| **`zero_cpu_matched`** | **0** | **1.1302** | **3.7814** | **17.266** | **8.823** | **none (3,600 upd)** | **—** | **0 / 0** | **0** | **0** |

**Key Findings (LBF-20D):**
1. **Fixed-exposure positive signal:** `bpd_svd` (`1.1688`, `-0.0563` regret vs `zero_0m`) outperformed `dense` (`1.1754`, `-0.0497`), `sham` (`1.2217`, `-0.0034`), and `zero/observe` (`1.2251`, exact observe parity confirmed).
2. **Capacity saturation at 4 MiB:** `4 MiB` and `16 MiB` produced identical hits (`19,376/30,720 = 63.1%`, 0 collisions) and identical regret (`1.1688`).
3. **Frozen learned BPD map (`--mms-map-manifest`) vs live SVD:** Fitting a 16-component sparse learned BPD map (`fit_bpd_map.py`, `bpd_recon_cosine=0.7859`, `1.55s` CPU, SHA-256 `66ef4c837b954167...`) from the trained checkpoint and holding it frozen from initialization **degraded** regret (`1.2444`), showing that frozen end-of-training maps misalign with early-training trajectories compared to live-refreshed decompositions.
4. **Fixed-CPU reversal (negative per-CPU-core-hour result):** Reinvesting the same CPU budget into plain training (`zero_cpu_matched`, `3,600` updates in `17.27s` CPU) achieved **`1.1302` regret**—beating `bpd_svd_4m` (`1.1688` in `22.89s` CPU) by `-0.0386` with 25% less CPU, because pure-Python `GradientBank` page lookup adds `~15x` CPU overhead per small-model step.

### B. LBF-20F (CSP Support Ranking, `updates=240`, `threads=2`, `seed=101`)

| Arm | `mms_mib` | Mean Nodes | Solved / 12 | Train CPU (s) | Wall (s) | Map Kind | Map Cos | Hits / Queries | Admitted | Alloc (B) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `zero_0m` | 0 | 3.08 | 11 (+1 refuted) | 2.219 | 1.142 | none | — | 0 / 0 | 0 | 0 |
| `obs_4m` | 4 | 3.08 | 11 (+1 refuted) | 5.359 | 2.683 | `svd_seed_proxy` | 0.9638 | 7,289 / 7,680 | 0 | 4,044,784 |
| `bpd_svd_4m` | 4 | 3.08 | 11 (+1 refuted) | 6.781 | 3.466 | `svd_seed_proxy` | 0.9647 | 7,289 / 7,680 | 6,923 | 4,044,784 |
| `bpd_svd_16m` | 16 | 3.08 | 11 (+1 refuted) | 10.859 | 5.572 | `svd_seed_proxy` | 0.9647 | 7,289 / 7,680 | 6,923 | 6,578,176 |
| `bpd_learned_16m` | 16 | 3.08 | 11 (+1 refuted) | 10.250 | 5.392 | `native_export` | 0.9518 | 7,289 / 7,680 | 6,892 | 6,578,176 |
| `dense_16m` | 16 | 3.08 | 11 (+1 refuted) | 10.031 | 5.304 | `svd_seed_proxy` | 0.9647 | 7,289 / 7,680 | 6,890 | 6,578,176 |
| `sham_16m` | 16 | **3.25** | 11 (+1 refuted) | 9.344 | 4.778 | `svd_seed_proxy` | 0.9637 | 7,289 / 7,680 | 7,061 | 6,578,176 |

- Learned BPD map (`fit_bpd_map.py`, `bpd_recon_cosine=0.9658`, `3.31s` CPU, SHA-256 `9b16d765b5f3e845...`) loaded cleanly via `--mms-map-manifest` (`map_cos=0.9518`).
- `94.9%` hit rate (`7,289/7,680`, 0 collisions), saturated at `mean_nodes=3.08` across all non-sham arms while `sham_16m` degraded to `3.25`; MMS added `2.4x–4.9x` CPU overhead.

### C. DataGoblin MOD7-MMS (`bijective`, `updates=1200`, `threads=2`, `seed=101`)

| Arm | `mms_mib` | `c6` (/512) | `ce6` | CPU (s) | Wall (s) | Map Kind | Map Cos | Hits / Queries | Admitted | Alloc (B) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `zero` | 0 | 80 | 1.9451 | 14.797 | 8.249 | none | — | 0 / 0 | 0 | 0 |
| `obs_4m` | 4 | 80 | 1.9451 | 106.484 | 57.233 | `svd_seed_proxy` | — | 17,940 / 115,200 | 0 | 4,194,000 |
| `dense_4m` | 4 | 77 | 1.9453 | 109.000 | 60.968 | `svd_seed_proxy` | — | 17,940 / 115,200 | 17,935 | 4,194,000 |
| `bpd_svd_4m` | 4 | 69 | 1.9462 | 107.438 | 57.605 | `svd_seed_proxy` | — | 17,940 / 115,200 | 17,939 | 4,194,000 |
| `bpd_learned_4m` | 4 | 79 | 1.9464 | 76.984 | — | `native_export` | 0.8633 | 17,940 / 115,200 | 10,554 | 4,194,000 |
| `sham_4m` | 4 | 80 | 1.9466 | 101.312 | 53.964 | `svd_seed_proxy` | — | 17,940 / 115,200 | 17,937 | 4,194,000 |

- Learned BPD map (`fit_bpd_map.py`, `bpd_recon_cosine=0.9088`, `1.86s` CPU, SHA-256 `d9099f950e7b1d38...`) reduced CPU vs live SVD (`76.98s` vs `107.44s` by avoiding periodic SVD refreshes), but all arms remain below the 256/512 competence gate and 5.2x–7.4x slower than `zero`.

### D. LBF-20M Exact Separator-Query Cache Lane (`--roots 3 --repeat 4`)

| `mms_mib` | Exact & Witness Valid | CPU Sum (s) | Wall Sum (s) | Peak Charged (B) | Evictions | Rejected | Root-0 Messages | Root-0 Bag Visits | Root-0 Hits / Misses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | True (100%) | 0.172 | 0.173 | 0 | 0 | 692 | 172 | 212 | 0 / 28 |
| 0.05 | True (100%) | **0.125** (`-27.3%`) | 0.131 | 11,804 | 0 | 0 | **127** (`-26.2%`) | **143** (`-32.5%`) | **24 / 4** (`85.7%`) |
| 0.25 | True (100%) | **0.125** (`-27.3%`) | 0.114 | 11,804 | 0 | 0 | **127** | **143** | **24 / 4** |
| 1.00 | True (100%) | **0.125** (`-27.3%`) | 0.127 | 11,804 | 0 | 0 | **127** | **143** | **24 / 4** |
| 4.00 | True (100%) | **0.125** (`-27.3%`) | **0.105** | 11,804 | 0 | 0 | **127** | **143** | **24 / 4** |

- **Strong positive CPU-efficiency result:** Exact certificate caching delivers **27.3% CPU savings** and **26.2% fewer separator messages** at just `11.5 KiB` charged payload (saturating immediately at `0.05 MiB`) while preserving 100% exactness and witness validity.

## 4. MMS-05: Utility-Directed Mechanistic Feedback Scheduler

Implemented in `src/datagoblin/training_gym/mms_utility.py` (`tests/test_dg_gym_mms_utility.py`, 3/3 unit tests passing, 0 extra forward/backward passes):
- Evaluates $U(s, r) = \frac{\mathbb{E}[\Delta V_{\mathrm{dev\text{-}train}} \mid s, r]}{\Delta C_{\mathrm{CPU}} + \lambda \Delta M}$ across typed domain utility signals (`lean_proof_progress`, `lbf_bellman_regret`, `mcp_retrieval_recovery`, `storyworld_repair`).
- Strictly rejects non-`dev_train` feedback partitions (`dev_final`, `confirmation`, `sealed`).
- Combines historical eligibility traces, online component gradient covariance SNR, byte-budgeted compressed physical-coordinate history, and cosine successful-repair retrieval; verified to reject high-hit/low-value sites that fool `hit_proxy`.

## 5. Conclusions & Promotion Disposition

1. **Exact RAM caching (`LBF-20M`) already wins per CPU core-hour** (`-27.3%` CPU, `-26.2%` messages, 100% certificate preservation).
2. **Gradient-replay MMS (`LBF-20D`) shows a real fixed-exposure gain** (`bpd_svd` `-0.0563` regret > `dense` `-0.0497` > `sham` `-0.0034` > `zero`), **but loses on fixed CPU** (`zero_cpu_matched` `1.1302` vs `bpd_svd` `1.1688`) due to `~15x` pure-Python `GradientBank` lookup overhead on small models.
3. **Prospective EvalMath confirmation is NOT yet justified** for neural gradient-replay MMS until either (a) the `GradientBank` hash/read/write path is vectorized/compiled to cut CPU overhead below `< 1.2x`, or (b) tested on larger models (`Mathlib TRM-XS`, `RMP` full-sequence rows) on a host with `> 12 GiB` free RAM and the staged `mathlib-c3` corpus.
