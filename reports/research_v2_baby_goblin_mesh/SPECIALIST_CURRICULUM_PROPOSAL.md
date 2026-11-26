# Baby Goblin Next-Stage Verifiers v2 Specialist Curriculum Proposal

**Target checkpoint lineage:** `AlephFunk/aleph-direct-a-10m-v2` (999,352,320 tied parameters; 14 unique blocks, 24 executed visits `[0, 1..10, 1..10, 11, 12, 13]`; frozen SmolLM2 tokenizer SHA-256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`).  
**Benchmark anchor:** Research Verifiers v2 (`implementation_sha256`: `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339`, commit `1385017a2da64e0248b9a4af5c49c0f2afdc16ac`).  
**Candidate dataset receipt:** [`training/research_v2_specialist/DATASET_RECEIPT.json`](../../training/research_v2_specialist/DATASET_RECEIPT.json) (`receipt_sha256`: `b7e6433eb84726dcd51b22ee7c120762dc130f79002a1ec2f7b92e53cbdbfdad`).

---

## 1. Empirical Motivation: Where Useful Gradient Signal Comes From

Our paired Research Verifiers v2 evaluations separate **three distinct failure regimes** that must be targeted in sequence rather than conflated into a single SFT pass:

1. **Pretraining-horizon deficit (Baby Goblin 10M v2):** At 10M tokens (mid-warmup, `lr=1.25e-5`, validation loss ~7.1 nats/token), Baby Goblin fails on **32/32** held cases by generating unigram punctuation/token loops up to the 512-token cap (`generation_truncated`) without emitting EOS or valid JSON. Native FP32 (`2e40e32d...`) and Q4_K_M GGUF (`c66b3756...`) agree 4/4 on `token_checks.json` and within `2.68e-6` KV-cache error, ruling out export corruption. **Implication:** Specialist SFT must only begin *after* Baby Goblin finishes enough base pretraining to reliably close JSON objects and emit EOS (`eos_id=0`); injecting specialist traces prematurely at 10M positions will overfit syntax onto under-trained core representations.
2. **Stateless-observation stopping & policy oscillation (observed in Qwen3-4B-Instruct-2507 Q8_0):** Even a 4.02B instruct model achieving **64/64** well-formed JSON responses, training real `AdamW` artifacts (`persona_externalization/72a94682c0a483f4`: `loss 1.113 -> 0.00173`, `train_acc=1.0`, `critical_fa=0/91`), and solving **2/8** symbolic tasks (`boolean_repair/670612341ca76f87`, `cursor_recovery/56624b5c277b4ea5`) to `check_passed=True` still scored **`0.0`** overall because:
   - Under Research v2's stateless `task-plus-complete-current-observation` history, once `check_passed=True` cleared the `error` field, deterministic greedy decoding oscillated back to `run_experiment` instead of emitting `submit`.
   - On learner tasks (`trm_training`, `ldt_filter`, `persona_externalization`), the model emitted schema-invalid config fields or repeated `steps=128` twice without a matched `steps=0` initialization baseline, then looped on `check` 6× despite `error: check unavailable; learned artifacts receive a final test after submission`, exhausting all 64 global actions in episodes 1–8 and starving episodes 9–32.
3. **Externalized persona boundary & opposite-frame invariance (Phase 4 Jinn/Beast):** On 432 unseen test roots (`MoralityLabAI/jinn-or-beast-paper@01512c40`), adding TRM frame-routing + 3-pass damped RLT correction on top of an external owner- and temporally-bound LDT head (198 parameters/frame: 75 LDT + 123 RLT) raises unseen-root policy accuracy from `97.69%` to **`99.31%`**, lifts Beast opposite-frame invariance from `95.14%` to **`100.0%`**, and holds critical-policy false accepts at **`0/155 (0.0%)`**, while ablating owner or temporal bindings causes catastrophic collapse (`100%` cross-owner leakage without owner binding; `3.94%` preference fidelity without temporal binding). Live in-context probes on 16 unseen test episodes confirmed that even `Qwen3-4B-Instruct-2507` Q8_0 (with `0/16` parse errors) achieves only `6.25%` preference fidelity and `0.0%` revocation handling in-context (`1/4` critical false accept after reset), proving persona state must be externalized rather than kept in base-model context.

---

## 2. Gradient-Signal Ranking Across All 8 Research v2 Families

| Priority | Task Family | Gradient Signal Strength | Primary Bottleneck Addressed | Recommended Curriculum Share |
|---:|---|---|---|---|
| **P0** | `persona_externalization` | **Highest** (dense 4-step trajectory: baseline `steps=0` -> fitted `steps>0` -> scoped memory reconstruction -> `submit`) | Externalized owner/scope/temporal/revocation state + matched `steps=0` baseline discipline + zero-FA critical veto | **30.0%** of synthetic tokens (plus private Jinn/Beast head intake) |
| **P0** | `trm_training` | **Highest** (exact optimizer config schema, `steps=0` vs `steps=128` paired delta, no illegal `check` calls) | Real optimizer loop control, baseline-vs-trained comparison, and immediate post-fit `submit` | **20.7%** |
| **P0** | `ldt_filter` | **Highest** (critical-veto enforcement, false-accept suppression, `steps=0` baseline pairing) | Evidence gating, `critical_veto=true` necessity, and avoiding schema hallucination | **19.9%** |
| **P1** | `cursor_recovery` | **High** (clean short arithmetic + explicit `check_passed=True -> submit` transition) | Partial-state retention (checkpoint cursor selection) and **post-check stopping** | **7.4%** |
| **P1** | `evidence_custody` | **High** (multi-record provenance filtering + `check -> repair -> submit`) | Quarantining unverified/unattested sources without silent omission | **7.9%** |
| **P1** | `experiment_design` | **Medium-High** (resource-constrained allocation + stopping after single verification) | Avoiding illegal `run_experiment` loops on analytical tasks | **5.6%** |
| **P2** | `boolean_repair` | **Medium** (counterexample witness vs equivalence + post-check stopping) | Counterexample search and immediate termination upon `check_passed=True` | **4.1%** |
| **P2** | `belief_update` | **Medium** (exact rational Bernoulli Bayes arithmetic) | Exact fraction normalization rather than approximate decimal heuristics | **4.3%** |

---

## 3. Staged Training Curriculum Design

### Stage 0 — Pretraining Gate (Prerequisite Before Specialist SFT)
- **Entry gate:** Continue base Baby Goblin pretraining past 10M positions until raw-completion JSON-closure rate on dev fixtures reaches `>= 90%` and EOS termination occurs well below 512 tokens (target: 500M–2B positions on the curated Aleph pretraining mix).
- **Frozen invariants:** Preserve the exact 999,352,320 tied-parameter Direct-A graph (`[0, 1..10, 1..10, 11, 12, 13]`) and SmolLM2 tokenizer (`9ca9acdd...`).

### Stage 1 — Failure-Directed TRM/LDT & Stateless-Stopping Specialist SFT
Train on the verified multi-turn transition records in `training/research_v2_specialist/candidate_train.jsonl`, masking loss strictly to the target JSON action (`supervised_tokens`) while conditioning on `task + current_observation`:

1. **Failure-Directed TRM/LDT Experiments (`59,632` total / `2,504` supervised Aleph tokens across `trm_training` + `ldt_filter`):**
   - Explicitly pairs a `steps=0` matched initialization run (`a1`) with a `steps=128` / `train_examples=256` AdamW run (`a2`), followed directly by `submit` with `claim: "improves"` — never calling `check` on learner tasks.
   - Injects explicit failure-recovery transitions (re-mapping invalid learner config keys or disallowed `check` errors directly to valid `run_experiment` / `submit` actions).
2. **Externalized Persona Control (`44,049` total / `1,348` supervised synthetic Aleph tokens + `1,850,745` total / `202,515` supervised private Jinn/Beast Aleph tokens):**
   - Teaches the model to maintain externalized owner/scope/temporal/revocation state outside transient context and to route policy checks to a small external LDT + TRM-routed RLT head (198 total parameters per frame: 75 base LDT + 123 RLT corrector) rather than relying on fragile in-context weight retention across model revisions.
   - Retains strict separation between the original Jinn/Beast test split (`432` roots), prospective Research v2 held cases (`32` roots), and specialist intake (`1,440` train roots).
3. **Stopping Decisions & Partial-State Retention (`42,960` total / `5,539` supervised Aleph tokens across `cursor_recovery`, `evidence_custody`, `experiment_design`, `boolean_repair`, `belief_update`):**
   - Directly eliminates the observed Qwen3 stateless-oscillation bug by including explicit state-transition targets:
     - `observation.status == "ready"` -> single valid `check` (on symbolic tasks) or `steps=0` `run_experiment` (on learner tasks).
     - `observation.check_passed == true` -> **mandatory immediate `submit`** (100% deterministic stopping rule, preventing 2-cycle `run_experiment <-> check` loops).
     - `observation.artifacts` contains `{baseline, trained}` -> **mandatory immediate `submit`** (preventing redundant identical `steps=128` runs and illegal `check` calls).
     - `cursor_recovery` / `evidence_custody` emphasize retaining verified partial state (exact checkpoint cursor positions, valid checksums, and quarantined record IDs) across stateless refreshes.

### Stage 2 — Online Verifier-Guided Reinforcement / Rejection-Sampling Fine-Tuning (RFT)
- Sample `N=8` candidate trajectories per **fresh train-only seed** against the isolated `ResearchSession` subprocess verifier (`RLIMIT`-capped).
- Admit trajectories only when `integrity_pass == True`, `primary_score >= 0.90`, `format_or_tool_errors == 0`, and (for learner tasks) `weights_changed == True` and `delta_from_initialization > 0`.
- Apply explicit negative weighting (or DPO pairs) against: (a) repeating an action after `error: check unavailable` or `error: experiment unavailable`, (b) omitting `steps=0` baselines, and (c) failing to call `submit` immediately after `check_passed == True`.

---

## 4. Verified Train-Only Candidate Dataset & Aleph Token Accounting

All synthetic records were generated from fresh train-only seeds (`generator_seed = 2026112601`), executed end-to-end through `ResearchSession` (including real AdamW learner fits and independent verification), deduplicated on both `root_id` and turn hash, and screened against **928 excluded roots and 928 excluded semantic digests** (`dev32-20261126`, `heldout-panel-20261126`, and `jinn-specialist-v2/held_roots.json`) with **0 overlaps**.

### 4.1 Public Train-Only Synthetic Corpus (`training/research_v2_specialist/candidate_train.jsonl`)

| Artifact / Field | Value |
|---|---|
| Dataset file | `training/research_v2_specialist/candidate_train.jsonl` (`808,445` bytes) |
| Dataset SHA-256 | `e03459a3205f03b9274dc1d8e314b7ee64d1ca5e75f286ab4787c29083673bf5` |
| Exclusion manifest | `training/research_v2_specialist/exclusion_manifest.json` (`df0984a60254b993faa4696db2fa8b7991bdc815ae57e2c97b3563a72b96ab54`) |
| Dataset receipt SHA-256 | `b7e6433eb84726dcd51b22ee7c120762dc130f79002a1ec2f7b92e53cbdbfdad` |
| Source rights | `Apache-2.0` (public & internal train-only) |
| Frozen Aleph tokenizer SHA-256 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| Independently verified episodes / unique roots | **128** (`16` per family across all 8 families; `0` failures) |
| Admitted multi-turn SFT records | **288** (`0` duplicate roots, `0` duplicate turns, `0` exclusion overlaps) |
| **Accepted Aleph tokens (total sequence)** | **146,641** |
| **Accepted Aleph tokens (supervised target)** | **9,391** |

#### Token Breakdown by Curriculum Bucket & Task Family

| Curriculum Bucket | Scope | Total Aleph Tokens | Share of Total |
|---|---|---:|---:|
| `failure_directed_trm_ldt` | `trm_training` (56 records, `30,378` tok) + `ldt_filter` (56 records, `29,254` tok) | 59,632 | 40.67% |
| `externalized_persona_control` | `persona_externalization` (56 records) | 44,049 | 30.04% |
| `partial_state_retention_and_stopping` | `cursor_recovery` (all turns) + turn-2 post-`check_passed=True` stopping transitions across symbolic families | 21,587 | 14.72% |
| `stopping_and_symbolic_verification` | Turn-1 verification/direct-submit transitions across `experiment_design`, `boolean_repair`, `belief_update`, `evidence_custody` | 21,373 | 14.58% |
| **Total (Public Synthetic Train-Only)** | **All 8 families (128 unique roots, 288 records; `9,391` supervised target tokens)** | **146,641** | **100.00%** |

Per-family exact Aleph token counts (`total` / `supervised target`): `persona_externalization` `44,049` / `1,348`; `trm_training` `30,378` / `1,288`; `ldt_filter` `29,254` / `1,216`; `evidence_custody` `11,578` / `962`; `cursor_recovery` `10,893` / `910`; `experiment_design` `8,162` / `2,088`; `belief_update` `6,340` / `1,162`; `boolean_repair` `5,987` / `417`.

### 4.2 Private Jinn/Beast Specialist Intake (Manifest-Only, Kept Outside Git)

| Field | Value |
|---|---|
| Source ID & revision | `jinn-beast-audited-tags-v1` (`01512c40be082ad88aa69594abca43a2b41908cc`) |
| Source rights & custody | Operator-authorized internal intake (`visibility: private`, `public_redistribution: unknown`; NEVER committed to Git) |
| Private `train.jsonl` SHA-256 | `d82d7204e745495ac60cb31ee7d53ccb96be4987412a9e0eec0811321c6e539f` (`8,479,055` bytes) |
| Private `held_roots.json` SHA-256 | `e43d654dc38a050247f79b1d1e67a5d52f5acb6a00b1d4e66b5c366c00746b84` (`896` excluded roots/semantics: 432 val + 432 test + 32 held) |
| Admitted unique train roots / targets | **1,440** roots / **2,880** targets (`1,440` Jinn + `1,440` Beast; all 4,608 source labels independently recomputed & verified) |
| **Accepted Aleph tokens (total sequence)** | **1,850,745** (`jinn`: `957,278`, `beast`: `893,467`) |
| **Accepted Aleph tokens (supervised target)** | **202,515** |

### 4.3 Combined Next-Stage Specialist Budget
- **Combined verified train-only pool:** **1,568** unique roots (`128` public synthetic + `1,440` private Jinn/Beast), **3,168** training records, **`1,997,386`** total accepted Aleph tokens (**`211,906`** supervised target tokens).
- **Recommended scale-up before Stage 1 training:** Expand the public synthetic generator from `16` to `256` verified train-only episodes per family (`2,048` roots -> ~`2.35M` Aleph tokens, ~`150k` supervised tokens) using the exact same `build_specialist_candidate_dataset.py` pipeline and exclusion manifest, keeping every future held panel strictly sealed and disjoint.

---

## 5. Promotion & Evaluation Protocol for Next Baby Goblin Checkpoints

1. Register each post-Stage-0 / post-Stage-1 checkpoint in a versioned registry via `register_checkpoint` with exact native safetensors SHA-256, `training_positions`, `training_index_sha256`, and `999,352,320` tied parameters (keeping GGUF arms registered under distinct IDs).
2. Tune learning rate, stopping-transition weight, and failure-recovery augmentation **only** on `dev32` fixtures.
3. Run **one** frozen held-panel evaluation per registered checkpoint on `heldout-panel-20261126` (`suite_sha256`: `cc87253ea6c9108def9e894cbed25957dd84a46928f2d13c271b3b4a39c292e1`) under identical budgets (`2,048` context, `512` max new tokens, `8` turns, `64` actions, `900s` wall cap, `temperature=0`, `seed=0`).
4. Gate promotion on:
   - `request_failures == 0` and `checkpoint_rank_eligible == True`;
   - `format_or_tool_errors == 0` across learner config schemas;
   - Positive `delta_from_initialization > 0` with `weights_changed == True` on `trm_training`, `ldt_filter`, and `persona_externalization`;
   - Zero post-`check_passed=True` oscillation loops (`submit` rate `== 100%` after verifier pass);
   - Zero critical-policy false accepts (`0.0%`) and `>= 99%` opposite-frame invariance on the externalized Jinn/Beast persona head.
