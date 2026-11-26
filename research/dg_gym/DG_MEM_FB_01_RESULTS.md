# DG-MEM-FB-01: Persistent RAM-Bounded Mechanistic Map & Localized BPD Defect Gym

Status: Public development only; local hardware, no paid compute.
Modules:
- `src/datagoblin/training_gym/mechanistic_map.py` (`MechanisticMapStore`, pure NumPy + stdlib, `mms_ram_budget_mb` enforced).
- `tests/test_dg_gym_mechanistic_map.py` (4 focused unit tests covering all 8 sections, hard byte caps, priority eviction, and `steering_feedback.select_window` integration).
- `research/dg_gym/dg_mem_fb_01.py` (35,399-parameter tied FiLM-gated recurrent reasoning model on a 4-regime verified modular-state task with localized BPD component defects).

## 1. Persistent Mechanistic Map (`mms_ram_budget_mb`)

`MechanisticMapStore(mms_ram_budget_mb)` enforces a strict byte cap `allocated_bytes <= round(mms_ram_budget_mb * 1024 * 1024)` across all eight mechanistic artifact families using deterministic lowest-priority, oldest-step eviction:

1. **BPD component decompositions & identities** (`BPDComponent`: `u`, `v`, `singular_value`, `weight_sha256`, `reconstruction_cosine`).
2. **Per-layer and per-visit causal contributions** (`CausalContribution`: accuracy delta, logit-margin delta, specificity).
3. **Activation statistics & steering directions** (`ActivationStats`: running mean, variance, norm mean, clean-vs-error steering direction).
4. **Jacobian & gradient-alignment estimates** (`JacobianGradAlignment`: component Jacobian norm, previous/base gradient cosine, directional sensitivity).
5. **Cross-pass transport maps** (`CrossPassTransport`: ridge-regressed `T_{v->v+1}` operators, auto-rank-compressed under tight `mms_ram_budget_mb` with `R^2` fidelity).
6. **Previous intervention outcomes** (`InterventionOutcome`: step, checkpoint SHA-256, target, pre/post accuracy, wall seconds, visit cost, acceptance).
7. **Component stability across optimizer updates** (`ComponentStability`: `cos(u_t, u_{t-1})`, `cos(v_t, v_{t-1})`, base drift, `should_reaudit` economic admission gate).
8. **Estimated marginal learning gains & intervention costs** (`MarginalGainEstimate`: `gain_per_visit`, wall/RAM cost bounds, bridged into `steering_feedback.select_window`).

## 2. Gym Design: Heterogeneous Localized BPD Defects

- **Model:** 35,399-parameter tied FiLM-gated recurrent reasoning block (`hidden_size=64`, `V=4` recurrent visits, 4 explicit linear sites `l0_up, l0_down, l1_up, l1_down`, top `R=16` BPD/SVD components per layer = 64 candidate components).
- **Verified Task:** 4-regime modular-state reasoning over $\mathbb{Z}_7$ (`add_shift`, `parity_step`, `threshold_gate`, `perm_hop`) with exact symbolic step traces and strictly disjoint `audit` (64) and `eval` (512) cohorts.
- **Defect Injection:** Corrupts only 4 localized BPD components (`l1_down:c0..c3`, 6.25% of the 64-component candidate space) with heterogeneous scaling while leaving `l0_up`, `l0_down`, `l1_up`, and `l1_down:c4..c15` intact.
- **Matched Budget:** `credit_cap = 160,000` visit-credits (1 per forward example-visit + 2 per backward example-visit), charging *all* mechanistic map builds, causal probes, and gradient updates to the same cap.

## 3. Observed Outcomes (All Negative Results Preserved)

### Campaign v3 (Competent Base: 2,200 pretrain steps, cosine LR, 3 seeds `11, 23, 37`, CPU 1-thread)

- **Base checkpoints:** mean eval accuracy **0.8405** (`0.8125` [`3e24a3084f0a25a3`], `0.8711` [`6d8a0eaa62900061`], `0.8379` [`aff8cef1fcb81491`]), well above `1/7 = 0.1429` chance.
- **After localized defect (`l1_down:c0..c3`):** mean eval accuracy drops to **0.1276** (`0.1230` [`ea1a5b4d2e1bdfaa`], `0.1309` [`24f29afc2c63834e`], `0.1289` [`d7957731741a8c48`]).
- **Process peak RAM:** `612.64 MB` (`receipt`: `research/dg_gym/receipts/dg_mem_fb_01_v3_receipt.json`).

| Arm | `mms_ram_budget_mb` | Mean Eval Acc | Gain vs `task_reinvest` | Defect Loc Recall | Map Alloc (B) | Map Evictions | Mean Wall (s) | Throughput (visits/s) | Paired (`n01`/`n10`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `task_reinvest` | 0.00 | 0.2253 | +0.0000 | 0.00 | 0 | 0.0 | 1.99 | 80,674 | 0 / 0 |
| `static_cyclic` | 0.00 | 0.2174 | -0.0078 | 0.00 | 0 | 0.0 | 2.88 | 55,627 | 73 / 85 |
| `shuffled_map` | 4.00 | 0.2038 | -0.0215 | 0.00 | 190,976 | 0.0 | 2.18 | 73,423 | 146 / 179 |
| `mms_guided_tiny` | 0.02 | 0.1973 | -0.0280 | 0.25 | 20,864 | 404.7 | 2.56 | 62,642 | 118 / 161 |
| `mms_guided_med` | 0.25 | 0.1986 | -0.0267 | 0.25 | 190,976 | 0.0 | 3.19 | 58,760 | 146 / 187 |
| `mms_guided_full` | 4.00 | 0.1986 | -0.0267 | 0.25 | 190,976 | 0.0 | 2.76 | 63,500 | 146 / 187 |

### Campaign v2 (Earlier 1,200-step Base, 3 seeds `11, 23, 37`, preserved negative pilot)

- Base mean accuracy **0.7077** → defect **0.1341** (`receipt`: `research/dg_gym/receipts/dg_mem_fb_01_v2_receipt.json`).
- `task_reinvest` **0.1901** beat `static_cyclic` (`-0.0059`), `shuffled_map` (`-0.0026`), `mms_guided_tiny` (`-0.0085`, `495.7` evictions), and `mms_guided_med/full` (`-0.0189`, `0` evictions), while `mms_guided` achieved `0.25` defect localization recall vs `0.00` for controls.

## 4. Scientific Interpretation

1. **Hard `mms_ram_budget_mb` control works as specified:** At `0.02 MB` (`20,971 B`), `MechanisticMapStore` stays strictly under budget (`20,864 B` allocated, `404.7` evictions) while preserving top-priority defective components (`0.25` localization recall matches the `4.00 MB` full map). At `0.25 MB` and `4.00 MB`, the full 8-section map (`190,976 B`) fits with zero evictions.
2. **Negative matched-budget repair result (extends `MOD7-FB-02`):** Even with a competent 84% base checkpoint and a genuinely localized 4-component BPD defect, `mms_guided` **fails to beat `task_reinvest`** under a matched `160,000` credit cap (`-0.0267` mean accuracy, negative on all 3 seeds). Corrupting the top 4 singular components of `l1_down` collapses all 4 recurrent visits to near-chance (`12.8%`), limiting single-component probe recall to `0.25` and making damped subspace-boosted AdamW slower than unmasked full-parameter AdamW (`task_reinvest`).
3. **Next hypothesis for DG-MEM-FB-02:** Test *regime-isolated* partial defects (where 3 of 4 regimes remain >85% intact while 1 regime drops to <30%) so full-parameter `task_reinvest` suffers catastrophic forgetting on healthy regimes unless guided by the persistent mechanistic map.
