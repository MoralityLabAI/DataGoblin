# Research Verifiers v2.1: Controller-Assisted Research, Externalized Persona Disentanglement, and Early Baby Goblin Specialization

**Status:** Partial Publication (Workstreams B & C 100% complete and verified; Workstream A v2.1 environment/adapter/tests/`vf-validate` and fresh `dev8-final` comparison complete; frozen 32-case prospective confirmation panel at `3/5` Qwen3 controller modes pending WSL recovery — incomplete denominators preserved verbatim).  
**Historical Baseline Preserved:** [PR #93](https://github.com/MoralityLabAI/DataGoblin/pull/93), commit `d4aa46c3ff68f03395cc0edadd5faae05782cd84`  
**v2.0 Frozen `implementation_sha256` (Untouched):** `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339`  
**v2.1 Frozen `implementation_sha256`:** `149e8082d3115d657880258bb2581f02fe1483d1da62adfee25a562e02b681f3`  
**Partial Ledger:** [`benchmark_v21_partial_ledger.json`](benchmark_v21_partial_ledger.json)  
**Documentation & Registry:** [`docs/research_verifiers_v2_1.md`](../../docs/research_verifiers_v2_1.md), [`configs/research_v2_1/models_v21.json`](../../configs/research_v2_1/models_v21.json)

---

## 1. Executive Summary

1. **Research Verifiers v2.1 (`datagoblin-research-v2.1`) is implemented, validated, and strictly separated from v2.0.** It adds per-episode action budgets, persistent `compact_state`, transparent intermediate `capability_score` reported separately from strict terminal `primary_score`, five auditable controller modes (`autonomous`, `deterministic_stopping`, `learned_ldt_stopping`, `trm_action_routing`, `rlt_state_correction`), and an independent declarative persona-store reference (`persona_store_reference.py`). All unit tests (`5/5`), `ruff` checks (`0` errors), and `vf-validate datagoblin-research-v2-1 --runtime.type subprocess -n 8` (`8/8` gold + `8/8` noop, `valid_rate = 1.0`) pass while leaving `datagoblin.research_v2` byte-identical.
2. **Workstream A (Dev8-Final & Partial Prospective-32):**
   - On 8 fresh development roots (`dev-v2.1-final`, `suite_sha256`: `f6e4010e...`), `qwen3-4b-instruct-2507-q8@rlt_state_correction` broke Qwen3's autonomous `0.0000` ceiling, reaching **`macro_score = 0.1401`** (solving `ldt_filter` end-to-end at **`99.61%`** held/OOD accuracy, `+27.73%` over matched zero-update baseline, `0` raw and guarded critical false accepts) and nearly doubling `macro_capability_score` (**`0.2688 -> 0.5038`**) while using **32% fewer inference calls** (`19` vs `28`, `303.9s` vs `443.6s`).
   - On the frozen 32-case prospective confirmation panel (`prospective-v2.1-20261127-final`, `suite_sha256`: `80a1dbc5...`, outside Git), `autonomous` (`9b8d0a26...`), `deterministic_stopping` (`f4c7f938...`), and `learned_ldt_stopping` (`e47b1d4d...`) completed before local WSL storage (`D:\`) filled to `10.3 MiB` free and halted `Ubuntu-24.04` during `trm_action_routing`. All completed and incomplete prospective denominators (`32/32` per run; `3/5` Qwen3 modes + `0/2` Baby Goblin modes completed so far) are preserved honestly; remaining prospective modes will complete on the same frozen panel once WSL recovers.
3. **Workstream B (Jinn/Beast 198-Parameter Hybrid Disentanglement — Complete):**
   - Multi-pass recurrence with **exact unrolled BPTT** contributes a **statistically significant, monotonic improvement** over both `fixed_mask_ldt` (75p) and `param_matched_feedforward` (198p) at identical parameter count on 864 unseen test targets (5 seeds): `98.26%` (`fixed_mask_ldt`) -> `98.66%` (`param_matched_feedforward`, not significant vs fixed-mask) -> `98.84%` (`K=2`) -> **`99.14%` (`K=3` exact, significant vs BOTH fixed-mask `[+0.23%, +1.50%]` and feedforward `[+0.12%, +0.93%]`)** -> **`99.58%` (`K=6` exact, `+1.27%` vs fixed-mask, `+0.93%` vs feedforward)**, while cutting raw (pre-veto) critical false accepts by >60% (`4.0 -> 1.4` combined) with `0` guarded critical false accepts across all arms.
   - Exact unrolled BPTT (`< 1e-6` central finite-difference error at `K=1, 3, 6`) is what pushes `K=3` past statistical significance over parameter-matched feedforward (`recurrent_3_pass_st` is `99.03%`, CI vs feedforward `[0.0%, +0.58%]`).
   - The independent declarative persona reference (`project_persona_store`) achieved **`432/432`** exact parity on unseen intervention episodes and **`8,000/8,000`** exact parity on randomized fuzz histories (`parity_rate = 1.0`).
4. **Workstream C (Early Baby Goblin 4-Arm Matched-Budget Specialist Experiment — Complete):**
   - Starting from exact native Baby Goblin 10M v2 (`2e40e32d...`, 999,352,320 tied parameters) using the exact **211,906 supervised-token** train-only pool (`9,391` Research v2 synthetic + `202,515` private Jinn/Beast, `0` excluded-root overlap) across 4 matched 128-update arms (~94.5k forward tokens/arm, `65,517,568` FP32 exit-shell trainable params over BF16 frozen visits `0..20`), **early specialization at 10M positions is not yet feasible**.
   - `arm3_base_then_specialist` (64 base + 64 specialist) strictly dominates `arm1_specialist_only` (128 specialist) at matched compute: lower action target NLL (`3.9823` vs `4.1202` nats), higher teacher-forced token accuracy (`11.05%` vs `7.89%`), higher `p(EOS)` (`0.0346` vs `0.0336`, >2,600× above `1.3e-5`), and **80% less general-language forgetting** (`+0.7198` vs `+3.5997` nats). Yet greedy autoregressive decoding still collapses to repeated JSON punctuation bigrams (`0/16` valid JSON, `0/16` EOS stops, `0.0` verified research progress across all 4 arms under both autonomous and RLT-assisted `ResearchSessionV21`).

---

## 2. Workstream A: Research Verifiers v2.1 Verification & Partial Controller Comparison

### 2.1 Environment, Adapter & Control Receipts

| Check / Control | Digest / Result |
|---|---|
| `datagoblin.research_v2` `implementation_digest()` | `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339` (unchanged from PR #81 / #93) |
| `datagoblin.research_v2_1` `implementation_digest()` | `149e8082d3115d657880258bb2581f02fe1483d1da62adfee25a562e02b681f3` |
| `pytest tests/unit/test_research_v2_1.py` | **5/5 passed** (includes exact BPTT `< 1e-5` finite-diff & persona tombstone resurrection) |
| `vf-validate datagoblin-research-v2-1 --runtime.type subprocess -n 8` | **8/8 gold + 8/8 noop valid (`valid_rate = 1.0`)** |
| Fresh `dev-v2.1-final` (16 cases, `f6e4010e...`) `control/reference` | `macro_score = 0.987305`, `macro_capability_score = 0.996826`, `success_rate = 1.0` (`319a8b77...`) |
| Fresh `dev-v2.1-final` (16 cases, `f6e4010e...`) `control/noop` | `macro_score = 0.000000`, `macro_capability_score = 0.000000`, `success_rate = 0.0` (`27b2c5bb...`) |
| Frozen `prospective-v2.1-20261127-final` (32 heldout cases outside Git) | `suite_sha256 = 80a1dbc545adec6dbc95842e8df5889decfe9e6674487956357085fa01c44be0`, `held_registry.json` SHA `f7d6f240...`, 0 overlap with v2.0 `dev32`/`held32`, `dev-v2.1`, Jinn/Beast, and specialist train |

### 2.2 Fresh Dev8-Final Qwen3 Oscillation Investigation (`dev-v2.1-final`, 8 Cases = 1/Family)

| Candidate Mode (`qwen3-4b-instruct-2507-q8`, CPU `--threads 10`) | Macro Primary | Macro Capability | Successes | Completed | Req Failures | Interventions (Skipped Turns) | Reqs | Wall (s) | Report SHA-256 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `autonomous` | `0.0000` | `0.2688` | `0/8` | `4/8` | `3` | `0 (0)` | `28` | `443.6` | `c6fb07ec...` |
| **`rlt_state_correction`** | **`0.1401`** | **`0.5038`** | **`1/8`** | **`6/8`** | `1` | **`9 (1)`** (`3` config sanitize, `2` zero-baseline, `2` check-exhaustion submit, `1` sym exp->check, `1` learner pair stop) | **`19`** | **`303.9`** | `37323d76...` |

*Key dev8-final finding:* Under `rlt_state_correction`, Qwen3 solved `ldt_filter/8b77cf6c22e22902` end-to-end (`primary_score = 0.9961`, `held_accuracy = 99.22%`, `ood_accuracy = 100.0%`, `delta_from_initialization = +0.2773`, `0` raw and guarded critical false accepts, `capability_score = 0.9990`), completed `trm_training/c2fcecf24fcbc312` with matched zero baseline and verified claim (`primary_score = 0.125`, `integrity_pass = True`, `capability_score = 0.7813`, preserving Qwen3's own chosen `lr=0.0001`), and ran a real 128-step fit on `persona_externalization` (`capability_score = 0.45`).

### 2.3 Frozen Prospective-32 Confirmation Panel (`prospective-v2.1-20261127-final`) — Partial Denominator Preserved

| Candidate Mode (`prospective-v2.1-20261127-final`, 32 cases) | Status | Macro Primary | Macro Capability | Successes | Req Failures | Interventions (Skipped) | Reqs | Wall (s) | Report SHA-256 |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `qwen3-4b-instruct-2507-q8@autonomous` | **Completed (32/32)** | `0.03125` | `0.17656` | `1/32` | `21` *(CPU 29s cap)* | `0 (0)` | `80` | `1392.2` | `9b8d0a26...` |
| `qwen3-4b-instruct-2507-q8@deterministic_stopping` | **Completed (32/32)** | `0.03125` | `0.12813` | `1/32` | `23` *(CPU 29s cap)* | `1 (1)` (`stop_after_verified_check`) | `83` | `1639.5` | `f4c7f938...` |
| `qwen3-4b-instruct-2507-q8@learned_ldt_stopping` | **Completed (32/32)** | `0.03125` | `0.13438` | `1/32` | `22` *(CPU 29s cap)* | `1 (1)` (`ldt_stop_after_verified_check`) | `83` | `1573.5` | `e47b1d4d...` |
| `qwen3-4b-instruct-2507-q8@trm_action_routing` | **Pending WSL recovery** | — | — | — | `Errno 30` (`D:\` full) | — | — | — | — |
| `qwen3-4b-instruct-2507-q8@rlt_state_correction` | **Pending WSL recovery** | — | — | — | Blocked (`D:\` full) | — | — | — | — |
| `baby-goblin-10m-v2-q4km@autonomous` | **Pending WSL recovery** | — | — | — | Blocked (`D:\` full) | — | — | — | — |
| `baby-goblin-10m-v2-q4km@rlt_state_correction` | **Pending WSL recovery** | — | — | — | Blocked (`D:\` full) | — | — | — | — |

*Incomplete denominator note:* No ranking is claimed across prospective modes until `trm_action_routing`, `rlt_state_correction`, and `baby-goblin-10m-v2-q4km` complete on the frozen panel (`80a1dbc5...`) after WSL recovery.

---

## 3. Workstream B: Jinn/Beast 198-Parameter Hybrid Disentanglement & Independent Persona Semantics (Complete)

**Script:** [`scripts/research_v2_1/jinn_beast_disentangle.py`](../../scripts/research_v2_1/jinn_beast_disentangle.py)  
**Verified Receipt:** [`receipts/jinn_beast_disentanglement.json`](receipts/jinn_beast_disentanglement.json) (`report_sha256`: `df92c3dc2d6153eac39c177880435e295c2f56afd074f35a591b9df13c5f7f6c`)  
**Split Integrity:** `1,440` train / `432` validation / `432` unseen test roots (`864` paired frame targets), `0` exclusion overlaps, all `4,608` source labels independently recomputed.  
**Exact Unrolled BPTT Verification (Central Finite Differences, $\varepsilon = 10^{-6}$):** `K=1: 5.461e-7`, `K=3: 7.982e-7`, `K=6: 1.098e-7` (all `< 1e-6`).  
**Independent Persona Reference Verification ([`persona_store_reference.py`](../../src/datagoblin/research_v2_1/persona_store_reference.py)):** `432/432` unseen intervention episodes exact match (`1,296` cross-owner, `432` cross-scope, `864` future-revision rejections; `216` active tombstones) + `2,000` randomized fuzz histories × `4` enforcement ablations = **`8,000/8,000`** exact parity (`parity_rate = 1.0`).

### 3.1 Full 12-Arm × 5-Seed Disentanglement Scorecard (864 Unseen Test Targets)

| Arm | Params / Frame | Guarded Acc (Combined) | Raw Acc (Combined) | Jinn Guarded / Raw | Beast Guarded / Raw | Raw Crit FA (Jinn / Beast) | Guarded Crit FA (J / B) | Opp-Frame Inv (J / B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `union_ldt_baseline` | 75 | 97.94% | 97.36% | 97.27% / 96.81% | 98.61% / 97.92% | 2.0 / 3.0 | **0 / 0** | 97.18% / 95.83% |
| `fixed_mask_ldt` | 75 | 98.26% | 97.80% | 98.15% / 97.92% | 98.38% / 97.69% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `learned_mask_ldt` *(L1 gates)* | 99 | 94.21% | 92.82% | 93.06% / 92.13% | 95.37% / 93.52% | 4.0 / 7.0 | **0 / 0** | 98.15% / 96.76% |
| `param_matched_feedforward` *(1-pass MLP)* | 198 | 98.66% | 98.19% | 98.15% / 97.92% | 99.17% / 98.47% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_1_pass_exact` | 198 | 98.66% | 98.19% | 98.15% / 97.92% | 99.17% / 98.47% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_1_pass_st` | 198 | 98.66% | 98.19% | 98.15% / 97.92% | 99.17% / 98.47% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_2_pass_exact` | 198 | 98.84% | 98.45% | 98.38% / 98.15% | 99.31% / 98.75% | 1.0 / 2.4 | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_2_pass_st` | 198 | 98.84% | 98.45% | 98.38% / 98.15% | 99.31% / 98.75% | 1.0 / 2.4 | **0 / 0** | **100.0% / 100.0%** |
| **`recurrent_3_pass_exact`** | **198** | **99.14%** | **98.82%** | **98.98% / 98.75%** | **99.31% / 98.89%** | **1.0 / 1.8** | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_3_pass_st` *(PR #93 ST)* | 198 | 99.03% | 98.70% | 98.75% / 98.52% | 99.31% / 98.89% | 1.0 / 1.8 | **0 / 0** | **100.0% / 100.0%** |
| **`recurrent_6_pass_exact`** | **198** | **99.58%** | **99.42%** | **99.31% / 99.21%** | **99.86% / 99.63%** | **0.4 / 1.0** | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_6_pass_st` | 198 | 99.54% | 99.31% | 99.31% / 99.07% | 99.77% / 99.54% | 1.0 / 1.0 | **0 / 0** | **100.0% / 100.0%** |

### 3.2 Paired Within-Root Bootstrap Contrasts (864 Test Targets, 2,000 Resamples)

| Contrast | Guarded Δ | Guarded 95% CI | Raw Δ | Raw 95% CI | Stat. Nonzero (95%) |
|---|---:|---|---:|---|---|
| `param_matched_feedforward` vs `fixed_mask_ldt` | `+0.35%` | `[0.00%, +0.81%]` | `+0.35%` | `[0.00%, +0.81%]` | No |
| `recurrent_2_pass_exact` vs `fixed_mask_ldt` | `+0.58%` | `[+0.12%, +1.16%]` | `+0.69%` | `[+0.23%, +1.39%]` | **Yes** |
| `recurrent_2_pass_exact` vs `param_matched_feedforward` | `+0.23%` | `[0.00%, +0.58%]` | `+0.35%` | `[0.00%, +0.81%]` | No |
| `recurrent_3_pass_st` vs `param_matched_feedforward` | `+0.23%` | `[0.00%, +0.58%]` | `+0.35%` | `[0.00%, +0.81%]` | No |
| **`recurrent_3_pass_exact` vs `param_matched_feedforward`** | **`+0.46%`** | **`[+0.12%, +0.93%]`** | **`+0.58%`** | **`[+0.12%, +1.16%]`** | **Yes (Both)** |
| **`recurrent_6_pass_exact` vs `fixed_mask_ldt`** | **`+1.27%`** | **`[+0.58%, +2.08%]`** | **`+1.62%`** | **`[+0.81%, +2.55%]`** | **Yes (Both)** |
| **`recurrent_6_pass_exact` vs `param_matched_feedforward`** | **`+0.93%`** | **`[+0.35%, +1.62%]`** | **`+1.27%`** | **`[+0.58%, +2.08%]`** | **Yes (Both)** |

**Empirical conclusion:** The recurrent component contributes a statistically significant, monotonic gain beyond both fixed-mask LDT and parameter-matched feedforward networks at identical 198-parameter budget, and exact unrolled BPTT strengthens both guarded accuracy and raw critical-policy false-accept suppression (`4.0 -> 1.4` combined raw FA at `K=6`).

---

## 4. Workstream C: Early Baby Goblin 4-Arm Matched-Budget Specialist Experiment (Complete)

**Script:** [`scripts/research_v2_1/baby_goblin_early_specialist_exp.py`](../../scripts/research_v2_1/baby_goblin_early_specialist_exp.py)  
**Verified Receipt:** [`receipts/baby_goblin_early_specialist.json`](receipts/baby_goblin_early_specialist.json) (`report_sha256`: `63a42e12d3096674c9cbdd1f9165c4f34f46890f8aa0404884e783a44d118e6e`)  
**Base Checkpoint:** `AlephFunk/aleph-direct-a-10m-v2` native FP32 (`2e40e32d...`, 999,352,320 tied parameters, 24-visit Direct-A schedule).  
**Specialist Pool:** Exact **211,906 supervised tokens** (`9,391` public Research v2 synthetic across 288 records + `202,515` private Jinn/Beast train-only across 2,880 records; `0` exclusion overlaps).  
**Compute & Scope:** Local RTX 5080 (idle, 0 leases; `3,356.5 MiB` peak VRAM, `10.5 GiB` peak RAM, `471.8s` wall), BF16 frozen prefix visits `0..20` (`933.8M` params) + FP32 trainable exit shell (`65,517,568` params: rank-16 LoRA on `blocks[11,12]` + full FP32 `block[13]` + `norm` + rank-64 `head_adapter`), FP32 `AdamW` (`lr=3e-4`, `wd=0.01`), **128 matched optimizer updates** (`~94.5k` forward tokens/arm).

| Arm (Matched 128 Updates) | Forward / Loss Tokens | Valid JSON | EOS Stop | Mean `p(EOS)` | Action Target NLL | TF Token Acc | General Val NLL (`Δ` vs 10M) | General PPL | `token_checks` | ResearchSessionV21 (`auto / RLT`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0_original_10m` (0u) | `0 / 0` | `0.0%` | `0.0%` | `1.3e-5` | `8.8126` | `0.00%` | `8.8218` (`0.0000`) | `6,780` | `4/4` | `0.0 / 0.0` |
| `arm1_specialist_only` (128 spec) | `94,357 / 13,993` | `0.0%` | `0.0%` | `0.0336` | `4.1202` | `7.89%` | `12.4215` (`+3.5997`) | `248,067` | `0/4` | `0.0 / 0.0` |
| `arm2_continued_base` (128 base) | `94,894 / 94,638` | `0.0%` | `0.0%` | `0.0000` | `10.7878` | `1.05%` | **`6.9570` (`-1.8648`)** | **`1,051`** | `2/4` | `0.0 / 0.0` |
| `arm3_base_then_specialist` (64+64) | `94,395 / 53,886` | `0.0%` | `0.0%` | **`0.0346`** | **`3.9823`** | **`11.05%`** | `9.5416` (`+0.7198`) | `13,927` | `0/4` | `0.0 / 0.0` |

**Empirical feasibility verdict:** Early specialization directly from Baby Goblin's 10M-position checkpoint is **not yet feasible**. Although `arm3_base_then_specialist` strictly outperforms `arm1_specialist_only` at matched compute (best action NLL `3.9823`, best TF token accuracy `11.05%`, >2,600× `p(EOS)` gain, and 80% less general-language forgetting), autoregressive greedy decoding still collapses to repeated JSON punctuation bigrams (`"{\"{\"..."` / `":\":\":..."`), yielding `0/16` valid JSON, `0/16` EOS terminations, and `0.0` verified research progress even with RLT controller assistance.
