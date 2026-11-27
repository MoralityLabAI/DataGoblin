# Research Verifiers v2.1: Controller-Assisted Research, Deep Recurrent Externalized Persona, and Staged Baby Goblin Emergence

**Status:** Complete Verified Publication (Workstream A frozen 32-case prospective controller comparison: `7/7` runs complete; Workstream B 12-arm disentanglement + 15-arm `K=1..24` deep recurrence & compute-matched FF controls on fresh 600-root confirmation panel complete; Workstream C early Baby Goblin 4-arm specialist pilot, forensic exposure/optimizer audit, and staged `20M/50M/100M/200M` emergence protocol complete).  
**Historical Baseline Preserved:** [PR #93](https://github.com/MoralityLabAI/DataGoblin/pull/93), commit `d4aa46c3ff68f03395cc0edadd5faae05782cd84`  
**v2.0 Frozen `implementation_sha256` (Untouched):** `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339`  
**v2.1 Frozen `implementation_sha256`:** `149e8082d3115d657880258bb2581f02fe1483d1da62adfee25a562e02b681f3`  
**Canonical Ledger:** [`benchmark_v21_ledger.json`](benchmark_v21_ledger.json) (`sha256: 75bd03f806c6048f5e6f5532914264d80d23c9cc5779ebd67b25568da6fdfc6a`)  
**Receipts Manifest:** [`receipts/RECEIPTS_MANIFEST.json`](receipts/RECEIPTS_MANIFEST.json) (`audit_issues: 0`)  
**Documentation & Registries:** [`docs/research_verifiers_v2_1.md`](../../docs/research_verifiers_v2_1.md), [`configs/research_v2_1/models_v21.json`](../../configs/research_v2_1/models_v21.json), [`docs/baby_goblin_staged_emergence_protocol.md`](../../docs/baby_goblin_staged_emergence_protocol.md), [`configs/research_v2_1/baby_goblin_emergence_ledger.json`](../../configs/research_v2_1/baby_goblin_emergence_ledger.json)

---

## 1. Executive Summary

1. **Research Verifiers v2.1 (`datagoblin-research-v2.1`) is cleanly versioned alongside an untouched v2.0 baseline.** It introduces per-episode action budgets, persistent `compact_state`, separate transparent `capability_score` alongside strict terminal `primary_score`, five auditable controller modes (`autonomous`, `deterministic_stopping`, `learned_ldt_stopping`, `trm_action_routing`, `rlt_state_correction`), and an independent declarative persona-store reference (`persona_store_reference.py`). All unit tests (`7/7`), `ruff` checks (`0` errors), and `vf-validate` checks for **both** `datagoblin-research-v2` and `datagoblin-research-v2-1` (`8/8` gold + `8/8` noop, `valid_rate = 1.0`) pass while leaving `datagoblin.research_v2` byte-identical.
2. **Workstream A (Frozen 32-Case Prospective Controller Comparison — 7/7 Runs Complete):**
   - On the sealed 32-case prospective panel (`prospective-v2.1-20261127-final`, `suite_sha256: 80a1dbc5...`, outside Git), stopping-only controllers (`deterministic_stopping`, `learned_ldt_stopping`) stay locked at Qwen3's autonomous ceiling (**`macro_score = 0.03125`**, `1/32` solved, `Δ = 0.0000`, saving `1` turn on the single verified `cursor_recovery` episode) because Qwen3 fails *upstream of verification* via learner config-schema errors and symbolic `run_experiment` calls.
   - **Action routing (`trm_action_routing`) and state correction (`rlt_state_correction`) both produce statistically nonzero paired 95% bootstrap gains over `autonomous` on BOTH strict `primary_score` and intermediate `capability_score`:**
     - **`qwen3@trm_action_routing`**: **`macro_score = 0.2280`** (**7.3×** autonomous; `Δ = +0.1968`, 95% CI `[+0.0989, +0.2998]`), **`macro_capability_score = 0.4929`** (**2.8×**; `Δ = +0.3164`, 95% CI `[+0.2042, +0.4328]`), **`7/32` solved** (`ldt_filter 3/4`, `cursor_recovery 2/4`, `boolean_repair 1/4`, `experiment_design 1/4`) + `2/4` `trm_training` integrity passes (`9` real optimizer fits, `14` persisted artifacts), cutting format/tool errors **93%** (`28 -> 2`), LLM calls (`80 -> 71`), wall time (`1392.2s -> 1125.9s`), and CPU 29s timeouts (`21 -> 9`).
     - **`qwen3@rlt_state_correction`**: **`macro_score = 0.1262`** (**4.0×** autonomous; `Δ = +0.0950`, 95% CI `[+0.0090, +0.1855]`), **`macro_capability_score = 0.3409`** (**1.9×**; `Δ = +0.1644`, 95% CI `[+0.0609, +0.2759]`), **`4/32` solved**, and the **only arm to solve `persona_externalization` end-to-end** on the prospective panel (`1/4`, `0.2236` family primary score, `1` externalized-persona intervention), cutting errors **96%** (`28 -> 1`) and inference calls (`80 -> 62`), though suffering `15` vs `9` host-side CPU 29s timeouts.
   - **Honest denominator preservation:** Because active `vramorama` GPU leases kept all prospective runs on CPU (`--threads 10 -ngl 0`), every run recorded nonzero 29s `TransportError` counts (`9–23` on Qwen3; `32/32` on Baby Goblin Q4_K_M whose `1.3e-5` EOS probability attempts all 512 completion tokens per turn). Accordingly, `checkpoint_rank_eligible = false` and `winner = null` are preserved verbatim for all 7 CPU runs; GPU non-timeout verification of Baby Goblin (`0/16` valid JSON, `0.0 / 0.0` on `ResearchSessionV21`) is separately documented in Workstream C.
3. **Workstream B (Jinn/Beast Disentanglement + Deep Recurrence `K=1..24`, Compute-Matched FF Controls & Fresh 600-Root Confirmation Panel — Complete):**
   - Treating the original 432 archive test roots as **SPENT** after the initial 12-arm study (`98.26% -> 99.58%`), we built a fresh, strictly disjoint **600-root (`1,200`-target)** confirmation panel outside Git (`panel_sha256: 6c491478...`, `0` overlap with all `2,304` archive roots) and verified exact unrolled BPTT against central finite differences (`0.0` relative error above `1e-7` tolerance) at `K = 1, 3, 6, 12, 24` tied and `depth = 6, 12` untied.
   - **Decisive empirical separation on the fresh 1,200-target confirmation panel (5-seed means):** Weight-tied recurrence with **only `198` parameters/frame** climbs monotonically from `K=3` (`75.88%`) -> `K=6` (`76.50%`) -> `K=12` (`76.73%`, statistically nonzero over `K=6`: `[+0.08%, +0.67%]`) -> **`K=24` (`76.80%`, best overall)** while slashing raw critical false accepts (`5.0 -> 2.0`, guarded `0/0`). Crucially, `198`-parameter tied recurrence **decisively outperforms BOTH compute-matched untied deep residual FF controls (`813–3,027` params, same `648–2,592` MACs, stalled flat at `74.55%–74.65%`, gap `+1.67%` to `+2.17%`, 95% CI strictly positive) AND compute-matched wide single-pass FF controls (`798–2,958` params, `75.07%–75.33%`, gap `+1.25%` to `+1.58%`, 95% CI strictly positive)**!
   - Exact unrolled BPTT beats straight-through approximation at every multi-pass depth (`K=6: 76.50%` vs `76.25%`; `K=12: 76.73%` vs `76.47%`; `K=24: 76.80%` vs `76.73%`), and convergence diagnostics reveal why untied deep MLPs fail: untied layers collapse updates onto the final layer (`tail_to_head_delta_ratio = 680.6–1,440.6`), whereas tied recurrence performs genuine iterative refinement across every pass (`per_pass_guarded_acc` rising smoothly from `74.5% -> 76.8%` and raw crit FA dropping `5 -> 2` across passes `1..24`).
   - Independent declarative persona reference ([`persona_store_reference.py`](../../src/datagoblin/research_v2_1/persona_store_reference.py)) achieved **`432/432`** unseen intervention + **`8,000/8,000`** fuzz parity (`1.0`).
4. **Workstream C (Early Baby Goblin 4-Arm Pilot, Forensic Exposure/Lineage Audit & Staged 20M/50M/100M/200M Protocol — Complete):**
   - Early specialization directly from Baby Goblin's 10M-position checkpoint (`2e40e32d...`) is **not yet feasible** (`0/16` valid JSON, `0/16` EOS stops, `0.0 / 0.0` on `ResearchSessionV21` across all 4 arms on RTX 5080), though `arm3_base_then_specialist` strictly dominates `arm1_specialist_only` at matched updates (`3.9823` vs `4.1202` target NLL, `11.05%` vs `7.89%` TF accuracy, `0.0346` vs `0.0336` `p(EOS)`, and **80% less forgetting**: `+0.72` vs `+3.60` nats).
   - Forensic exact replay audit ([`baby_goblin_exposure_audit.json`](receipts/baby_goblin_exposure_audit.json), `receipt_exact_replay_verified: true` on all 4 arms) exposed four structural causes: (i) **6.76× forward-vs-loss token asymmetry** (`94,638` vs `13,993` loss tokens between `arm2` and `arm1`), (ii) **sparse specialist coverage** (`6.60%` of the 211,906-token pool in `arm1`, `3.21%` in `arm3`), (iii) **prompt left-truncation at `max_seq_len=384`** (`100.0%` of Jinn/Beast prompts, dropping `263.6` leading tokens; `63.89%` of synthetic prompts; `0` targets truncated), and (iv) **optimizer lineage discontinuity** (weights-only export restarted with fresh FP32 AdamW at `lr=3e-4` = **24×** the 10M end-of-warmup `lr=1.25e-5`, freezing `93.44%` of parameters).
   - We published the complete staged **20M / 50M / 100M / 200M** pretraining & specialist-adaptation protocol ([`docs/baby_goblin_staged_emergence_protocol.md`](../../docs/baby_goblin_staged_emergence_protocol.md)), machine-readable emergence ledger ([`configs/research_v2_1/baby_goblin_emergence_ledger.json`](../../configs/research_v2_1/baby_goblin_emergence_ledger.json)), and reusable emergence probe ([`scripts/research_v2_1/baby_goblin_emergence_probe.py`](../../scripts/research_v2_1/baby_goblin_emergence_probe.py)) to record the exact first checkpoint where JSON syntax, EOS termination, action selection, and verified research progress emerge.

---

## 2. Workstream A: Research Verifiers v2.1 Verification & Complete Prospective Controller Comparison

### 2.1 Environment, Adapter & Deterministic Control Receipts

| Check / Control | Digest / Result |
|---|---|
| `datagoblin.research_v2` `implementation_digest()` | `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339` (unchanged from PR #81 / #93) |
| `datagoblin.research_v2_1` `implementation_digest()` | `149e8082d3115d657880258bb2581f02fe1483d1da62adfee25a562e02b681f3` |
| `pytest tests/unit/test_research_v2_1.py` | **7/7 passed** (exact BPTT `K=1..24` tied + `depth=6,12` untied, persona tombstone resurrection, controllers, emergence ledger) |
| `vf-validate datagoblin-research-v2 --runtime.type subprocess -n 8` | **8/8 gold + 8/8 noop valid (`valid_rate = 1.0`)** |
| `vf-validate datagoblin-research-v2-1 --runtime.type subprocess -n 8` | **8/8 gold + 8/8 noop valid (`valid_rate = 1.0`)** |
| Fresh `dev-v2.1-final` (16 cases, `f6e4010e...`) `control/reference` | `macro_score = 0.987305`, `macro_capability_score = 0.996826`, `success_rate = 1.0` (`319a8b77...`) |
| Fresh `dev-v2.1-final` (16 cases, `f6e4010e...`) `control/noop` | `macro_score = 0.000000`, `macro_capability_score = 0.000000`, `success_rate = 0.0` (`27b2c5bb...`) |
| Frozen `prospective-v2.1-20261127-final` (32 heldout cases outside Git) | `suite_sha256 = 80a1dbc545adec6dbc95842e8df5889decfe9e6674487956357085fa01c44be0`, `0` overlap with v2.0 `dev32`/`held32`, `dev-v2.1`, Jinn/Beast, and specialist train |

### 2.2 Fresh Dev8-Final Qwen3 Oscillation Investigation (`dev-v2.1-final`, 8 Cases = 1/Family)

| Candidate Mode (`qwen3-4b-instruct-2507-q8`, CPU `--threads 10`) | Macro Primary | Macro Capability | Successes | Completed | Req Failures | Interventions (Skipped Turns) | Reqs | Wall (s) | Report SHA-256 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `autonomous` | `0.0000` | `0.2688` | `0/8` | `4/8` | `3` | `0 (0)` | `28` | `443.6` | `c6fb07ec...` |
| **`rlt_state_correction`** | **`0.1401`** | **`0.5038`** | **`1/8`** | **`6/8`** | `1` | **`9 (1)`** (`3` config sanitize, `2` zero-baseline, `2` check-exhaustion submit, `1` sym exp->check, `1` learner pair stop) | **`19`** | **`303.9`** | `37323d76...` |

### 2.3 Complete Frozen Prospective-32 Confirmation Scorecard (`prospective-v2.1-20261127-final`, 7/7 Runs)

All 7 prospective runs executed on CPU (`--threads 10 -ngl 0`) to leave active `vramorama` GPU leases undisturbed, with identical per-episode budgets (`max_turns = 6`, `max_experiments = 4`, `context = 2,048`, `max_tokens = 512`, `per_request_timeout = 29s`). Because every CPU run incurred transport timeouts (`request_failures > 0`), `checkpoint_rank_eligible = false` and `winner = null` are preserved strictly across all runs.

| Candidate Mode (`prospective-v2.1-20261127-final`, 32 cases) | Macro Primary | Macro Capability | Successes | Integrity Passes | Real Fits (Artifacts) | Tool / Format Errors | Req Failures (29s CPU cap) | Interventions (Skipped Turns) | Reqs | Wall (s) | Peak RAM (MiB) | Report SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `qwen3-4b@autonomous` | `0.03125` | `0.17656` | `1/32` | `1` | `0 (0)` | `28` | `21` | `0 (0)` | `80` | `1392.2` | `5895` | `9b8d0a26...` |
| `qwen3-4b@deterministic_stopping` | `0.03125` | `0.12813` | `1/32` | `1` | `0 (0)` | `45` | `23` | `1 (1)` (`stop_after_verified_check`) | `83` | `1639.5` | `5890` | `f4c7f938...` |
| `qwen3-4b@learned_ldt_stopping` | `0.03125` | `0.13438` | `1/32` | `1` | `0 (0)` | `44` | `22` | `1 (1)` (`ldt_stop_after_verified_check`) | `83` | `1573.5` | `5661` | `e47b1d4d...` |
| **`qwen3-4b@trm_action_routing`** | **`0.22803`** | **`0.49294`** | **`7/32`** | **`9`** | **`9 (14)`** | **`2`** | **`9`** | **`34 (6)`** (`9` sanitize, `8` sym->chk, `6` chk-submit, `5` zero-base, `3` chk-stop, `3` pair-stop) | `71` | **`1125.9`** | `5503` | `4fd50e84...` |
| **`qwen3-4b@rlt_state_correction`** | **`0.12622`** | **`0.34093`** | **`4/32`** | **`5`** | **`5 (8)`** | **`1`** | `15` | **`24 (4)`** (`6` chk-submit, `6` sym->chk, `5` sanitize, `3` zero-base, `2` chk-stop, `2` pair-stop; **`1` ext. persona**) | **`62`** | `1190.9` | `6055` | `5815f7ea...` |
| `baby-goblin-q4km@autonomous` | `0.00000` | `0.00000` | `0/32` | `0` | `0 (0)` | `0` | `32` *(no EOS)* | `0 (0)` | `32` | `938.9` | `2254` | `8310deba...` |
| `baby-goblin-q4km@rlt_state_correction` | `0.00000` | `0.00000` | `0/32` | `0` | `0 (0)` | `0` | `32` *(no EOS)* | `0 (0)` | `32` | `939.1` | `2310` | `04601ff6...` |

### 2.4 Paired Within-Family Bootstrap Contrasts vs `qwen3-4b@autonomous` (32 Prospective Cases)

| Candidate vs `qwen3-4b@autonomous` | Primary Δ | Primary 95% CI | Capability Δ | Capability 95% CI | Key Per-Family Primary Gains (`4` cases/family) |
|---|---:|---|---:|---|---|
| `qwen3-4b@deterministic_stopping` | `+0.0000` | `[+0.0000, +0.0000]` | `-0.0484` | `[-0.0984, +0.0031]` | Identical `cursor_recovery 1/4`; saved 1 model turn after verified check |
| `qwen3-4b@learned_ldt_stopping` | `+0.0000` | `[+0.0000, +0.0000]` | `-0.0422` | `[-0.0922, +0.0047]` | Identical `cursor_recovery 1/4`; saved 1 model turn after verified check |
| **`qwen3-4b@trm_action_routing`** | **`+0.1968`** | **`[+0.0989, +0.2998]`** | **`+0.3164`** | **`[+0.2042, +0.4328]`** | **`ldt_filter +0.75 (3/4)`, `cursor_recovery +0.25 (2/4)`, `boolean_repair +0.25 (1/4)`, `experiment_design +0.25 (1/4)`, `trm_training +0.074 (2/4 integrity)`** |
| **`qwen3-4b@rlt_state_correction`** | **`+0.0950`** | **`[+0.0090, +0.1855]`** | **`+0.1644`** | **`[+0.0609, +0.2759]`** | **`persona_externalization +0.224 (1/4 — ONLY arm to solve)`, `ldt_filter +0.25 (1/4)`, `experiment_design +0.25 (1/4)`, `trm_training +0.036 (1/4 integrity)`** |
| `baby-goblin-q4km@autonomous` | `-0.0313` | `[-0.0938, +0.0000]` | `-0.1766` | `[-0.2375, -0.1156]` | All `32/32` Turn-1 timeouts at 29s CPU cap (`0` completions; GPU non-timeout `0.0/0.0` in Section 4) |
| `baby-goblin-q4km@rlt_state_correction` | `-0.0313` | `[-0.0938, +0.0000]` | `-0.1766` | `[-0.2375, -0.1156]` | All `32/32` Turn-1 timeouts at 29s CPU cap (`0` completions; controller cannot act without response) |

*Interpretation of `trm_action_routing` vs `rlt_state_correction`:* Both action-routing and RLT state-correction achieve **statistically significant gains over autonomous execution** on strict `primary_score` and `capability_score` while lowering total inference calls and wall time. `rlt_state_correction` is the **only arm that solved a sealed `persona_externalization` episode end-to-end** (`0.2236` family mean, `externalized_persona_interventions = 1`), whereas `trm_action_routing` attained higher overall prospective primary score (`0.2280` vs `0.1262`) primarily because CPU latency jitter near the 29s hard cap caused `15` Turn-1/mid-episode timeouts during `rlt_state_correction` versus `9` during `trm_action_routing`.

---

## 3. Workstream B: Jinn/Beast Disentanglement, Deep Recurrence (`K=12, 24`), Compute-Matched FF Controls & Fresh 600-Root Confirmation

### 3.1 Phase 1: 12-Arm × 5-Seed Disentanglement & Independent Persona Semantics (Archive 432 Test Roots — Now Spent)

**Script:** [`scripts/research_v2_1/jinn_beast_disentangle.py`](../../scripts/research_v2_1/jinn_beast_disentangle.py)  
**Verified Receipt:** [`receipts/jinn_beast_disentanglement.json`](receipts/jinn_beast_disentanglement.json) (`report_sha256: df92c3dc2d6153eac39c177880435e295c2f56afd074f35a591b9df13c5f7f6c`)  
**Independent Persona Reference ([`persona_store_reference.py`](../../src/datagoblin/research_v2_1/persona_store_reference.py)):** `432/432` unseen intervention episodes exact parity (`1,296` cross-owner, `432` cross-scope, `864` future-revision rejections; `216` active tombstones) + `2,000` randomized fuzz histories × `4` enforcement ablations = **`8,000/8,000`** exact parity (`parity_rate = 1.0`).  
**Exact Unrolled BPTT Verification ($\varepsilon = 10^{-6}$):** `K=1: 5.461e-7`, `K=3: 7.982e-7`, `K=6: 1.098e-7` (all `< 1e-6`).

| Arm (Original Archive Split: `1,440` Train / `432` Val / `432` Test = 864 Targets — **Spent Reference**) | Params / Frame | Guarded Acc (Comb.) | Raw Acc (Comb.) | Raw Crit FA (J / B) | Guarded Crit FA (J / B) | Opp-Frame Inv (J / B) |
|---|---:|---:|---:|---:|---:|---:|
| `union_ldt_baseline` | 75 | 97.94% | 97.36% | 2.0 / 3.0 | **0 / 0** | 97.18% / 95.83% |
| `fixed_mask_ldt` | 75 | 98.26% | 97.80% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `learned_mask_ldt` *(L1 gates)* | 99 | 94.21% | 92.82% | 4.0 / 7.0 | **0 / 0** | 98.15% / 96.76% |
| `param_matched_feedforward` *(1-pass MLP)* | 198 | 98.66% | 98.19% | 1.0 / 3.0 | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_2_pass_exact` / `_st` | 198 | 98.84% | 98.45% | 1.0 / 2.4 | **0 / 0** | **100.0% / 100.0%** |
| **`recurrent_3_pass_exact`** | **198** | **99.14%** | **98.82%** | **1.0 / 1.8** | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_3_pass_st` *(PR #93 ST)* | 198 | 99.03% | 98.70% | 1.0 / 1.8 | **0 / 0** | **100.0% / 100.0%** |
| **`recurrent_6_pass_exact`** | **198** | **99.58%** | **99.42%** | **0.4 / 1.0** | **0 / 0** | **100.0% / 100.0%** |
| `recurrent_6_pass_st` | 198 | 99.54% | 99.31% | 1.0 / 1.0 | **0 / 0** | **100.0% / 100.0%** |

### 3.2 Phase 2: Deep Recurrence (`K=12, 24`), Compute-Matched FF Controls, Convergence Diagnostics & Fresh 600-Root Confirmation Panel

**Script:** [`scripts/research_v2_1/jinn_beast_depth_study.py`](../../scripts/research_v2_1/jinn_beast_depth_study.py)  
**Verified Receipt:** [`receipts/jinn_beast_deep_recurrence_study.json`](receipts/jinn_beast_deep_recurrence_study.json) (`report_sha256: 7c87bf4934b9a39c973c867470fc25140cce80ed0d6f23f222ace7e8377d9b46`, `386.1s` wall)  
**Fresh Independent Confirmation Panel (Stored Strictly Outside Git):** `600` fresh roots (`1,200` paired Jinn/Beast frame targets, `panel_sha256: 6c4914783f65effa416758545eec4357d91aff8823430329dcbb9548a6106aec`, `seed = 2026112788`), **`0` overlap** with all `2,304` archive roots (including the `432` **spent** test roots) and external exclusions.  
**Exact Unrolled BPTT Verification ($\varepsilon = 10^{-6}$, abs tol $10^{-7}$):** `0.0` relative error across `tied K = 1, 3, 6, 12, 24` AND `untied depth = 6, 12`.

#### 3.2.1 Full 15-Arm × 5-Seed Scorecard on Fresh 600-Root (1,200-Target) Confirmation Panel

| Arm | Params / Frame | Corrector MACs | Fresh Guarded Acc (1,200) | Fresh Raw Acc (1,200) | Fresh Raw Crit FA (J / B) | Fresh Guarded Crit FA | Spent 432 Ref. Guarded | Tail/Head Δ Ratio (B / J) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_mask_ldt` | 75 | 0 | 73.82% | 73.73% | 0.0 / 5.0 | **0 / 0** | 98.26% | — |
| `param_matched_ff_1x` | 198 | 108 | 74.67% | 74.58% | 0.0 / 4.2 | **0 / 0** | 98.66% | 1.00 / 1.00 |
| `recurrent_3_pass_exact` | **198** | 324 | 75.88% | 75.80% | **0.0 / 2.0** | **0 / 0** | 99.14% | 3.33 / 3.24 |
| **`recurrent_6_pass_exact`** | **198** | **648** | **76.50%** | **76.42%** | **0.0 / 2.0** | **0 / 0** | 99.58% | 7.03 / 6.63 |
| `recurrent_6_pass_st` | 198 | 648 | 76.25% | 76.17% | 0.0 / 2.0 | **0 / 0** | 99.54% | 6.93 / 6.63 |
| `compute_matched_ff_depth6` *(untied 6-layer)* | 813 | 648 | 74.65% | 74.57% | 0.0 / 4.2 | **0 / 0** | 98.66% | **285.7 / 136.8** |
| `compute_matched_ff_wide6` *(wide 1-pass)* | 798 | 648 | 75.07% | 74.98% | 0.0 / 3.0 | **0 / 0** | 98.73% | 1.00 / 1.00 |
| **`recurrent_12_pass_exact`** | **198** | **1,296** | **76.73%** | **76.65%** | **0.0 / 2.0** | **0 / 0** | **99.65%** | 14.36 / 13.29 |
| `recurrent_12_pass_st` | 198 | 1,296 | 76.47% | 76.38% | 0.0 / 2.0 | **0 / 0** | 99.54% | 14.16 / 13.35 |
| `compute_matched_ff_depth12` *(untied 12-layer)* | 1,551 | 1,296 | 74.55% | 74.47% | 0.0 / 4.2 | **0 / 0** | 98.61% | **680.6 / 352.4** |
| `compute_matched_ff_wide12` *(wide 1-pass)* | 1,518 | 1,296 | 75.17% | 75.08% | 0.0 / 2.4 | **0 / 0** | 98.73% | 1.00 / 1.00 |
| **`recurrent_24_pass_exact`** | **198** | **2,592** | **76.80%** | **76.72%** | **0.0 / 2.0** | **0 / 0** | **99.65%** | 28.95 / 26.55 |
| `recurrent_24_pass_st` | 198 | 2,592 | 76.73% | 76.65% | 0.0 / 2.0 | **0 / 0** | 99.61% | 28.63 / 26.75 |
| `compute_matched_ff_depth24` *(untied 24-layer)* | 3,027 | 2,592 | 74.62% | 74.53% | 0.0 / 4.4 | **0 / 0** | 98.63% | **1,206.6 / 1,440.6** |
| `compute_matched_ff_wide24` *(wide 1-pass)* | 2,958 | 2,592 | 75.33% | 75.25% | 0.0 / 2.0 | **0 / 0** | 98.84% | 1.00 / 1.00 |

#### 3.2.2 Paired Within-Root Bootstrap Contrasts on Fresh 1,200 Confirmation Targets (2,000 Resamples)

| Primary Contrast (Fresh 600-Root / 1,200-Target Panel) | Guarded Δ | Guarded 95% CI | Raw Δ | Raw 95% CI | Stat. Nonzero (95%) |
|---|---:|---|---:|---|---|
| `recurrent_6_pass_exact` (198p) vs `compute_matched_ff_depth6` (813p) | **`+1.67%`** | **`[+1.00%, +2.42%]`** | `+1.67%` | `[+1.00%, +2.42%]` | **Yes** |
| `recurrent_6_pass_exact` (198p) vs `compute_matched_ff_wide6` (798p) | **`+1.25%`** | **`[+0.67%, +1.92%]`** | `+1.25%` | `[+0.67%, +1.92%]` | **Yes** |
| `recurrent_12_pass_exact` (198p) vs `compute_matched_ff_depth12` (1,551p) | **`+2.17%`** | **`[+1.42%, +3.00%]`** | `+2.17%` | `[+1.42%, +3.00%]` | **Yes** |
| `recurrent_12_pass_exact` (198p) vs `compute_matched_ff_wide12` (1,518p) | **`+1.58%`** | **`[+0.92%, +2.33%]`** | `+1.58%` | `[+0.92%, +2.33%]` | **Yes** |
| **`recurrent_12_pass_exact` vs `recurrent_6_pass_exact`** | **`+0.33%`** | **`[+0.08%, +0.67%]`** | `+0.33%` | `[+0.08%, +0.67%]` | **Yes** |
| `recurrent_24_pass_exact` (198p) vs `compute_matched_ff_depth24` (3,027p) | **`+2.08%`** | **`[+1.33%, +2.92%]`** | `+2.08%` | `[+1.33%, +2.92%]` | **Yes** |
| `recurrent_24_pass_exact` (198p) vs `compute_matched_ff_wide24` (2,958p) | **`+1.42%`** | **`[+0.83%, +2.08%]`** | `+1.42%` | `[+0.83%, +2.08%]` | **Yes** |
| `recurrent_24_pass_exact` vs `fixed_mask_ldt` (75p) | **`+2.92%`** | **`[+2.00%, +3.92%]`** | `+2.92%` | `[+2.00%, +3.92%]` | **Yes** |
| `recurrent_24_pass_exact` vs `recurrent_12_pass_exact` | `+0.00%` *(seed0; 5-seed mean `+0.07%`)* | `[+0.00%, +0.00%]` | `+0.00%` | `[+0.00%, +0.00%]` | Saturation (`K=12 -> 24`) |

#### 3.2.3 Convergence Diagnostics: Why Weight-Tied Recurrence Beats 15× Larger Compute-Matched MLPs

- **Genuine multi-step refinement vs last-layer collapse:** In `recurrent_24_pass_exact` (seed 0, Beast), raw update norms remain well-conditioned across all 24 passes (`1.034 -> 1.248`), guarded accuracy rises smoothly across passes (`74.50%` at pass 1 -> `75.67%` at pass 9 -> `76.33%` at pass 18 -> `76.83%` at pass 22–24), and raw critical false accepts drop monotonically (`5 -> 4` at pass 6 -> `3` at pass 9 -> `2` at pass 15–24). By contrast, `compute_matched_ff_depth24` (`3,027` params) learns near-zero updates in layers `1..22` (`damped_delta_l2 = 0.0003` to `0.0110`) and dumps `>90%` of its update into layer 24 (`0.4344`, `tail_to_head_delta_ratio = 1,206.6`), leaving passes `1..20` flat at `74.33%` and raw critical FA stuck at `5`.
- **Exact BPTT vs Straight-Through gradient propagation:** At `K=24` (epoch 40), exact BPTT propagates strong upstream credit all the way to pass 1 (`||dcur||` rises smoothly from `0.0052` at pass 24 to `0.0080` at pass 1 on Beast, `0.0043 -> 0.0129` on Jinn), driving final training loss `2.15×` lower (`0.00282` vs `0.00607`), whereas straight-through exponentially attenuates early passes by $\alpha_j = 1/(K - j)$ (`0.00650 -> 0.00027`, a **24× drop**).
- **Depth scaling law:** Recurrence gains are statistically significant from `K=1 -> K=3 -> K=6 -> K=12` (`74.67% -> 75.88% -> 76.50% -> 76.73%`), then saturate gently between `K=12` and `K=24` (`76.73% -> 76.80%`), making **`K=12` exact BPTT** the Pareto-optimal operating point (`198` params, `1,296` MACs).

---

## 4. Workstream C: Early Baby Goblin 4-Arm Pilot, Forensic Exposure/Lineage Audit & Staged Emergence Protocol

### 4.1 Four-Arm Matched-Budget Experiment (10M Native Checkpoint, Local RTX 5080)

**Script:** [`scripts/research_v2_1/baby_goblin_early_specialist_exp.py`](../../scripts/research_v2_1/baby_goblin_early_specialist_exp.py)  
**Verified Receipt:** [`receipts/baby_goblin_early_specialist.json`](receipts/baby_goblin_early_specialist.json) (`report_sha256: 63a42e12d3096674c9cbdd1f9165c4f34f46890f8aa0404884e783a44d118e6e`)  
**Base Checkpoint:** `AlephFunk/aleph-direct-a-10m-v2` native FP32 (`2e40e32d...`, 999,352,320 tied parameters, 24-visit Direct-A schedule).  
**Specialist Pool:** Exact **211,906 supervised tokens** (`9,391` public Research v2 synthetic across 288 records + `202,515` private Jinn/Beast train-only across 2,880 records; `0` exclusion overlaps).  
**Compute & Scope:** Local RTX 5080 (`3,356.5 MiB` peak VRAM, `10.5 GiB` peak RAM, `471.8s` wall), BF16 frozen prefix visits `0..20` (`933.8M` params) + FP32 trainable exit shell (`65,517,568` params = `6.56%`), FP32 `AdamW` (`lr=3e-4`, `wd=0.01`), **128 matched optimizer updates** (`~94.5k` forward tokens/arm).

| Arm (Matched 128 Updates) | Forward / Loss Tokens | Valid JSON | EOS Stop | Mean `p(EOS)` | Action Target NLL | TF Token Acc | General Val NLL (`Δ` vs 10M) | General PPL | `token_checks` | ResearchSessionV21 (`auto / RLT`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0_original_10m` (0u) | `0 / 0` | `0.0%` | `0.0%` | `1.3e-5` | `8.8126` | `0.00%` | `8.8218` (`0.0000`) | `6,780` | `4/4` | `0.0 / 0.0` |
| `arm1_specialist_only` (128 spec) | `94,357 / 13,993` | `0.0%` | `0.0%` | `0.0336` | `4.1202` | `7.89%` | `12.4215` (`+3.5997`) | `248,067` | `0/4` | `0.0 / 0.0` |
| `arm2_continued_base` (128 base) | `94,894 / 94,638` | `0.0%` | `0.0%` | `0.0000` | `10.7878` | `1.05%` | **`6.9570` (`-1.8648`)** | **`1,051`** | `2/4` | `0.0 / 0.0` |
| `arm3_base_then_specialist` (64+64) | `94,395 / 53,886` | `0.0%` | `0.0%` | **`0.0346`** | **`3.9823`** | **`11.05%`** | `9.5416` (`+0.7198`) | `13,927` | `0/4` | `0.0 / 0.0` |

### 4.2 Forensic Token-Exposure & Optimizer-Lineage Audit

**Script:** [`scripts/research_v2_1/audit_baby_goblin_exposure.py`](../../scripts/research_v2_1/audit_baby_goblin_exposure.py)  
**Verified Receipt:** [`receipts/baby_goblin_exposure_audit.json`](receipts/baby_goblin_exposure_audit.json) (`audit_sha256: 55fc1d54ac4a1761fa079ff105ff6f2fda3b7632ab469871927e2673579566c7`, `receipt_exact_replay_verified: true` on all 4 arms)

| Audit Dimension | Measured Value Across 4-Arm Pilot | Implication for Staged 20M–200M Protocol |
|---|---|---|
| **Forward-vs-loss token asymmetry** | `arm1` (`94,357` fwd / `13,993` loss, ratio `0.1483`) vs `arm2` (`94,894` fwd / `94,638` loss, ratio `0.9973`) — **6.76×** more loss-bearing tokens in `arm2` | Track and report **both** forward positions and supervised loss tokens; run `>= 2` full specialist epochs (`>= 423,812` loss tokens) |
| **Specialist pool coverage** | `arm1`: `6.60%` of 211,906 pool (`106/288` synth, max 4× repeat; `123/2,880` Jinn/Beast, max 2× repeat). `arm3`: `3.21%` (`53/288` synth; `64/2,880` Jinn/Beast + `47,085` base tokens) | Deterministic epoch shuffling over 100% of the train-only specialist pool paired with 50% general replay |
| **Prompt left-truncation (`max_seq_len=384`)** | **`100.0%` (`2,880/2,880`)** of Jinn/Beast prompts truncated (mean raw `576.3 -> 312.7`, dropping `263.6` tokens); **`63.89%` (`184/288`)** of synthetic prompts truncated (`476.6 -> 317.4`); `0` targets truncated | Train all staged `20M–200M` forks at full **`2,048`-token context** so policy headers and `compact_state` stay 100% untruncated |
| **Optimizer lineage discontinuity** | Native export is weights-only (no pretraining AdamW moments); pilot froze `933.8M` BF16 prefix (`93.44%`) and ran fresh FP32 AdamW at `lr=3e-4` (**24×** 10M end-of-warmup `lr=1.25e-5`; peak `2.5e-4` at 20M) on `65.5M` exit shell (`6.56%`) | Resume pretraining milestones from durable `group_039` checkpoints (weights + FP32 AdamW moments + RNG + cursor); fork specialist runs at `lr=5e-5` cosine |

### 4.3 Staged `20M / 50M / 100M / 200M` Emergence Protocol & Milestone Ledger

- **Protocol Specification:** [`docs/baby_goblin_staged_emergence_protocol.md`](../../docs/baby_goblin_staged_emergence_protocol.md)
- **Machine-Readable Emergence Ledger:** [`configs/research_v2_1/baby_goblin_emergence_ledger.json`](../../configs/research_v2_1/baby_goblin_emergence_ledger.json) (observed `10M` arms populated; all 7 emergence milestones currently `null`; clean slots for `20M`, `50M`, `100M`, `200M`)
- **Executable Probe Runner:** [`scripts/research_v2_1/baby_goblin_emergence_probe.py`](../../scripts/research_v2_1/baby_goblin_emergence_probe.py) (evaluates any native Aleph Direct-A safetensors checkpoint on the frozen 16-case non-held emergence suite `eval_seed = 2026112799`, updates the ledger, and records the first checkpoint crossing: `first_json_onset >= 0.25`, `first_json_reliable >= 0.80`, `first_eos_onset >= 0.25`, `first_eos_reliable >= 0.80`, `first_action_selection >= 0.50`, `first_verified_capability >= 0.25`, and `first_nonzero_primary_score > 0.0`).
