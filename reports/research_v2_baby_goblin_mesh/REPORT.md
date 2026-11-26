# Research Verifiers v2 Empirical Benchmark: Baby Goblin 10M v2 vs. Compact-Model Research Mesh

**Benchmark Suite:** `MoralityLabAI/DataGoblin` Research Verifiers v2 (PR #81, commit `1385017a2da64e0248b9a4af5c49c0f2afdc16ac`, `implementation_sha256`: `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339`)  
**Frozen Held Panel:** 32 cases (4 per family across all 8 families), stored outside Git (`suite_sha256`: `cc87253ea6c9108def9e894cbed25957dd84a46928f2d13c271b3b4a39c292e1`, `held_registry.json` SHA-256: `441bde1496389e5b402eba970e2556df4a25a81366e22a758ffe16f887b7a9a6`)  
**Frozen Comparison Contract:** `comparison_sha256` `a487b866b49a9b1512413bf8e18b4e86c9ffd4c72839eec35d74de815766f3cd` (`context_tokens=2048`, `max_new_tokens_per_turn=512`, `max_turns=8`, `effective_max_actions=64`, `effective_run_seconds=900`, `temperature=0.0`, `seed=0`, `history="task-plus-complete-current-observation"`, per-turn timeout `30.0s`, fixed Linux `RLIMIT` learner/verifier sandbox)  
**Immutable Benchmark Ledger:** [`benchmark_ledger.json`](benchmark_ledger.json) (`ledger_sha256`: `26ba9ef908564995ee92eb412ceeed76c3aae7766b29ade5ab67298750f1670a`)  
**Specialist Proposal & Train-Only Dataset:** [`SPECIALIST_CURRICULUM_PROPOSAL.md`](SPECIALIST_CURRICULUM_PROPOSAL.md) & [`DATASET_RECEIPT.json`](../../training/research_v2_specialist/DATASET_RECEIPT.json)

---

## 1. Executive Verdict

1. **Baby Goblin 10M v2 cannot yet conduct useful research experiments (`macro_score = 0.0`, `0/32` successes, `32/32` `generation_truncated`).** Both the native FP32 checkpoint (`2e40e32d...`, 999,352,320 tied parameters, 24 executed block visits) and its Q4_K_M GGUF deployment arm (`c66b3756...`) pass strict local parity checks (`4/4` `token_checks.json` top-1 & token-ID matches, `2.68e-6` KV-cache parity), ruling out export/serving bugs. The failure is strictly the **10M-position pretraining horizon** (mid-warmup, loss ~7.1 nats/token): Baby Goblin emits repetitive unigram punctuation without EOS until hitting the 512-token turn limit on all 32 held cases.
2. **Pretrained 4B baselines expose a distinct research-policy bottleneck rather than pure syntax failure.** On the identical 32-case held panel:
   - **`qwen3-4b-instruct-2507-q8`** (`Qwen/Qwen3-4B-Instruct-2507` Q8_0 GGUF, CPU, **0 transport failures / 64 of 64 valid JSON replies**) executed **2 real 128-step `AdamW` learner fits** (`loss 1.113 -> 0.00173`, `100%` train acc, `0/91` critical false accepts) and solved **2 of 8 attempted cases** (`boolean_repair/670612341ca76f87`, `cursor_recovery/56624b5c277b4ea5`) to `check_passed = True` twice each — yet still scored **`0.0` overall** because stateless refreshed observations induced a deterministic greedy 2-cycle oscillation (`run_experiment <-> check`) that never called `submit` and exhausted all 64 actions in the first 8 episodes.
   - **`qwen3-4b`** (pinned `Qwen/Qwen3-4B@1cfa9a72`, BF16, `thinking_mode=disabled`, CPU) **did call `submit`** after `experiment unavailable` on 2 of 4 `evidence_custody` cases (`completed = 2/32`), demonstrating cleaner stopping behavior than the 2507 instruct arm, but hit the 29s CPU per-turn deadline on `26/32` turns (`macro_score = 0.0`) while the workstation RTX 5080 was leased by concurrent external `vramorama` jobs.
3. **No model ranking is claimed (`winner: null`, `checkpoint_rank_eligible: false` for all evaluated models).** Paired `runner.compare()` across the 3 complete-denominator 32-case held runs records `macro_delta = 0.0` (95% within-family paired bootstrap CI `[0.0, 0.0]`) because none of the evaluated checkpoints achieved `checkpoint_rank_eligible == True` and GPU-bound recurrent baselines (`dory-4b-bf16-loop2`, `nanbeige42-3b-loop2`, `hrm-text-1b-h2-l3`, `inky-mini-varloop-research-pass3`) remain deferred pending a free GPU lease.
4. **Externalizing persona state into a bound LDT + TRM-routed RLT head decisively outperforms base-model in-context reasoning (Phase 4, Jinn/Beast).** On 432 unseen Jinn/Beast test roots (`MoralityLabAI/jinn-or-beast-paper@01512c40`), live `Qwen3-4B-Instruct-2507` Q8_0 in-context achieved only **`6.25%` preference fidelity** and **`0.0%` revocation handling** (`50.0%` routed policy accuracy; `0/16` parse errors) on full shuffled event histories and suffered a **`25.0%` (1/4) critical-policy false accept** after context reset. Our external owner- and temporally-bound LDT + TRM routing + 3-pass RLT head (198 total parameters/frame: 75 LDT + 123 RLT) achieved **`100.0%` preference fidelity**, **`100.0%` temporal revision correctness**, **`100.0%` revocation handling**, **`0.0%` cross-owner leakage**, **`100.0%` opposite-frame invariance** on both Jinn and Beast, **`99.31%` unseen-root policy accuracy**, and **`0/155 (0.0%)` guarded critical-policy false accepts**.

---

## 2. Phase 1 Environment & Private Artifact Verification

| Check | Status | Verified Digest / Count |
|---|---|---|
| Environment install (`-e '.[dev]' -e environments/datagoblin_research_v2`, Linux WSL) | **PASS** | `verifiers 0.4.1`, `numpy 2.5.5`, Linux `RLIMIT` active |
| `implementation_digest()` integrity | **PASS** | `73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339` (unchanged from PR #81) |
| PR #81 unit + integration tests | **PASS** | `156 passed` (`test_research_v2*`, `test_research_v2_verifiers.py`) |
| `vf-validate datagoblin-research-v2 --runtime.type subprocess -n 8` | **PASS** | `8/8` gold valid, `8/8` noop valid (`valid_rate = 1.0`) |
| Dev32 deterministic `control/reference` | **PASS** | `macro_score = 0.989014`, `success_rate = 1.0`, 24 fitted artifacts, mean learner delta `+0.312826`, `report_sha256` `965ade55...` |
| Dev32 deterministic `control/noop` | **PASS** | `macro_score = 0.0`, `success_rate = 0.0`, `report_sha256` `739527a5...` |
| Private Jinn/Beast archive (`MoralityLabAI/jinn-or-beast-paper@01512c40`) | **PASS** | Exact SHA-256 match on all 3 pinned files (`2a4594c6...`, `85e544c3...`, `8828fbe4...`); all 4,608 labels recomputed; 2,880 train targets / 1,440 roots (`train.jsonl` `d82d7204...`, kept outside Git) |
| Private Research v2 Held Panel (`heldout-panel-20261126`) | **PASS** | 32 fresh cases outside Git (`suite_sha256` `cc87253e...`), 0 overlap with dev32 or Jinn/Beast (`896` excluded roots) |

---

## 3. Candidate Registry & Complete Benchmark Scorecard

All six requested candidate families plus local quantized deployment arms are registered in [`configs/research_v2/models_mesh_20261126.json`](../../configs/research_v2/models_mesh_20261126.json). Sanitized binding manifests are archived in [`bindings/`](bindings/).

### 3.1 Held-Panel (32-Case) & Control Scorecard

| Candidate / Control | Format & Serving | Actual Params | Recurrence | Held32 Status | Macro Score | Successes | Completed | Req Failures | Rank Eligible | Report SHA-256 |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| `control/reference` *(dev32)* | Deterministic gold | — | — | Control (dev32) | **0.9890** | 32/32 | 32/32 | 0 | No (control) | `965ade551e03...` |
| `control/noop` *(dev32)* | Deterministic noop | — | — | Control (dev32) | **0.0000** | 0/32 | 0/32 | 0 | No (control) | `739527a579da...` |
| **`baby-goblin-10m-v2-q4km`** | Q4_K_M GGUF (`llama-server` CPU) | 999,352,320 *(1.67B stored)* | 10 core × 2 (24 visits) | **Evaluated (32/32)** | **0.0000** | 0/32 | 0/32 | 32 *(trunc)* | No | `fd646d39f99b...` |
| **`qwen3-4b-instruct-2507-q8`** | Q8_0 GGUF (`llama-server` CPU) | 4,022,468,096 | None | **Evaluated (64/64 acts)** | **0.0000** | 0/32 | 0/32 | **0** | No | `f95a65e909f7...` |
| **`qwen3-4b`** (`think=False`) | Native BF16 (`transformers` CPU) | 4,022,468,096 | None | **Evaluated (32 reqs)** | **0.0000** | 0/32 | **2/32** | 26 *(timeout)* | No | `bf5a0c4cd913...` |
| `baby-goblin-10m-v2-native` | Native FP32 (`safetensors`) | 999,352,320 | 10 core × 2 (24 visits) | **Parity verified; CPU 30s blocked** | — | — | — | — | Blocked | Evaluated via `q4km` arm |
| `dory-4b-bf16-loop2` | Native BF16 (`transformers`) | 3,975,229,440 | `n_recurrent_loops=2`, `per_loop` | **Loaded; GPU leased / CPU >30s** | — | — | — | — | Blocked | — |
| `nanbeige42-3b-loop2` | Native BF16 (`transformers`) | **4,169,800,704** *(3B non-emb)* | `num_loops=2` | **Loaded; GPU leased / CPU >30s** | — | — | — | — | Blocked | — |
| `hrm-text-1b-h2-l3` | Native BF16 (PrefixLM adapter) | 1,182,795,264 | `H2/L3` (8× inner) | **Loaded; GPU leased / CPU >30s** | — | — | — | — | Blocked | — |
| `qwen3-4b` (`think=True`) | Native BF16 (`transformers`) | 4,022,468,096 | None | **CPU smoke >29s (`<think>`)** | — | — | — | — | Blocked | — |
| `inky-mini-varloop-research-pass3` | Native `nvfp4-mxfp8-bf16` (+ IQ4_NL GGUF) | 26,388,646,440 *(~5.4B active)* | `num_passes=3` | **No native runtime / >16 GiB VRAM** | — | — | — | — | Blocked | — |

### 3.2 Checkpoint Identity & Local Verification Ledger

| Registry ID | Source Revision | Verified Weights SHA-256 | Tokenizer / Config Verification | Hardware / Blocking Note |
|---|---|---|---|---|
| `baby-goblin-10m-v2-native` | `9ac6fe2e...` | `2e40e32da6a6d975...` (3.997 GB) | SmolLM2 `9ca9acdd...`; `4/4` `token_checks.json` match; `2.68e-6` KV diff | ~3.5 tok/s CPU FP32 (~146s for 512 toks > 30s cap) while RTX 5080 leased |
| `baby-goblin-10m-v2-q4km` | `9ac6fe2e...` | `c66b3756710fe71a...` (1.030 GB) | Frozen raw completion (`no-cnv`); 32/32 held cases completed without timeout | Evaluated full held32 on CPU (~38 tok/s) |
| `qwen3-4b` | `1cfa9a72...` | Index SHA `6abb9b5d...` (3 shards) | `config` `8ba006f7...`, `tok_cfg` `d5d09f07...`, 4,022,468,096 params | Evaluated full held32 (`think=False`) on CPU; `think=True` >29s |
| `qwen3-4b-instruct-2507-q8` | `cdbee75f...` | `c892d3128b07169d...` (4.28 GB) | `config` `5beea1a4...`, `tok_cfg` `a62ff0a2...`, 4,022,468,096 params | Evaluated full held32 + dev8 on CPU (0 timeouts) |
| `dory-4b-bf16-loop2` | `ee517dd3...` | `207483664dfef54c...` (7.951 GB) | `config` `d2d086f5...`, `tok_cfg` `0f5c10ba...`, native `dory` load 3,975,229,440 params | Loaded in 18.0s (8.26 GiB RAM); 2-loop BF16 CPU >30s + external GPU lease |
| `nanbeige42-3b-loop2` | `b82e54bd...` | Index SHA `503dbea9...` (2 shards) | `config` `f6cb15b2...`, `tok_cfg` `3edfa64a...`, 4,169,800,704 actual params | Manual `DynamicCache` decode fixed `transformers 5.24.1` `cache` kwarg error; CPU >29s |
| `hrm-text-1b-h2-l3` | `22097cbc...` | `f8fe2b2bf6948414...` (2.366 GB) | `config` `5a708f12...`, `tok_cfg` `260d448d...`, PrefixLM all-ones `token_type_ids`, 1,182,795,264 params | `direct` & `synth,cot` bindings built; 8× H2/L3 CPU >28.5s |
| `inky-mini-varloop-research-pass3` | `91a57fe2...` | IQ4_NL `e9f70a16...` (17.63 GB) | `config` `32a0abdf...`, `tok_cfg` `bf09342d...`, 26.39B total / ~5.4B active | Native `nvfp4-mxfp8-bf16` (20.2 GB) lacks local runtime & exceeds 16 GiB VRAM; Drive-mounted IQ4_NL timed out >240s startup |

*Workstation compute note:* During the entire evaluation window, the local RTX 5080 Laptop GPU (16 GiB VRAM) was occupied by external concurrent `vramorama` leases (`gym-b25-train-onebit-bdb` / `squid5080-serve` using 14.1–15.6 GiB VRAM with `gym-b25-bench-onebi` queued). Per policy, no competing jobs were disturbed and no paid external accelerators were provisioned.

---

## 4. Per-Family Performance, Real Training Activity & Resource Efficiency

### 4.1 Per-Family Outcomes Across Evaluated Held32 Runs & Dev32 Reference Control

| Family (4 held / 4 dev cases) | `control/reference` (dev32) | `baby-goblin-10m-v2-q4km` (held32) | `qwen3-4b-instruct-2507-q8` (held32) | `qwen3-4b` BF16 `think=False` (held32) |
|---|---:|---:|---:|---:|
| `trm_training` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, 8 tool errs, 0 fits) | 0.000 (0/4, 4 timeouts) |
| `ldt_filter` | **0.961** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, 8 tool errs, 0 fits) | 0.000 (0/4, 4 timeouts) |
| `persona_externalization` | **0.951** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, **2 real AdamW fits**, 6 tool errs) | 0.000 (0/4, 4 timeouts) |
| `experiment_design` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, 8 tool errs) | 0.000 (0/4, 4 timeouts) |
| `boolean_repair` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, **2/2 checks passed**, 0 submit) | 0.000 (0/4, 3 timeouts, 1 budget) |
| `belief_update` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, 0/2 checks passed, 6 errs) | 0.000 (0/4, 3 timeouts, 1 budget) |
| `evidence_custody` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, 0/2 checks passed, 6 errs) | 0.000 (0/4, **2/4 submitted**, 2 errs) |
| `cursor_recovery` | **1.000** (4/4) | 0.000 (0/4, 4 trunc) | 0.000 (0/4, **2/2 checks passed**, 0 submit) | 0.000 (0/4, 0/1 check passed, 2 timeouts) |
| **Macro / Total** | **0.9890 (32/32)** | **0.0000 (0/32)** | **0.0000 (0/32; 4 checks passed, 2 fits)** | **0.0000 (0/32; 2 submitted)** |

### 4.2 Rollout Accounting, Verifier Calls & Resource Consumption (Held32)

| Metric | `baby-goblin-10m-v2-q4km` | `qwen3-4b-instruct-2507-q8` | `qwen3-4b` (BF16 `think=False`) |
|---|---:|---:|---:|
| Inference requests / HTTP requests | 32 / 64 | 64 / 128 | 32 / 64 |
| Successful inference replies | 0 | **64** | 6 |
| Transport failures (`generation_truncated` / `timeout`) | 32 (`32` trunc, `0` timeout) | **0** | 26 (`25` worker deadline, `1` request deadline) |
| Session action/schema errors | 0 | 54 | 3 |
| Actions emitted (`run_experiment` / `check` / `submit`) | 0 / 0 / 0 | 33 / 31 / **0** | 3 / 1 / **2** |
| Real optimizer experiments / persisted artifacts | 0 / 0 | **2 / 2** (`256` updates, `65,792` cell calls) | 0 / 0 |
| Verifier `check` calls (passed / failed) | 0 (0 / 0) | **8** (**4 passed** / 4 failed) | 1 (0 / 1) |
| Budget-exhausted unattempted episodes | 0 | 24 *(first 8 eps used all 64 actions)* | 4 *(900s wall cap)* |
| Prompt tokens / Completion tokens | 10,708 / 16,384 | 21,303 / 2,550 | 2,389 / 270 *(26 calls unknown due to timeout)* |
| Wall time / CPU time / GPU time | 440.3s / 3,242.6s / 0.0s | 423.9s / 3,168.3s / 0.0s | 900.1s / 6,772.9s / 0.0s |
| Load time / Peak RAM / Peak VRAM | 2.31s / 2,853.6 MiB / 0 MiB | 3.08s / 5,635.2 MiB / 0 MiB | 19.14s / 8,572.0 MiB / 0 MiB |

### 4.3 Paired Differences & Uncertainty Intervals (`runner.compare()`)

Using `baby-goblin-10m-v2-q4km` as baseline across the exact 32 paired held roots (`suite_sha256` `cc87253e...`, `interval_status = "paired_within_family"`, 2,000 bootstrap resamples):
- `qwen3-4b-instruct-2507-q8` vs `baby-goblin-10m-v2-q4km`: `macro_delta = 0.0000`, 95% CI `[0.0000, 0.0000]`, `rank_eligible = False`.
- `qwen3-4b` vs `baby-goblin-10m-v2-q4km`: `macro_delta = 0.0000`, 95% CI `[0.0000, 0.0000]`, `rank_eligible = False`.
- `winner`: `null` (neither candidate crossed eligibility or score > 0; functional behavioral differences appear cleanly in rollout accounting, verifier passes, and learner artifact telemetry below).

---

## 5. Failure Taxonomy & Causal Separation

Our verified transcripts cleanly separate **three orthogonal root causes**:

### 5.1 Pretraining Horizon (Baby Goblin 10M v2)
- **Evidence:** Native FP32 (`2e40e32d...`) and Q4_K_M GGUF (`c66b3756...`) match 4/4 on `token_checks.json` and within `2.68e-6` KV-cache parity, proving the 24-visit Direct-A core repetition and raw-completion prompt serialization execute faithfully.
- **Concrete failure:** Across all 32 held episodes (e.g., `trm_training/3619b09983a0e7b9`, 295 prompt tokens -> 512 completion tokens in `15.55s`), Baby Goblin never emits EOS (`eos_id=0`) or JSON syntax, repeating high-frequency unigram punctuation (`,,,,...`) until hitting `finish_reason = "length"` (`generation_truncated`). This is an expected consequence of stopping pretraining at 10M positions (loss ~7.1 nats/token, mid-warmup), not an export defect.

### 5.2 Research Policy & Stateless Stopping Failures (Qwen3-4B Family)
Concrete independently verified episodes from `runs_held32/qwen3-4b-instruct-2507-q8` and `runs_held32/qwen3-4b`:
1. **Verified real AdamW training without baseline pairing (`persona_externalization/72a94682c0a483f4`, `qwen3-4b-instruct-2507-q8`):**
   - Turn 1 (`req=17`): Valid `run_experiment` (`steps=128, train_examples=256, learning_rate=0.5, critical_veto=true`) persisted artifact `a1` (`artifact_sha256` `878c5651...`, 75 params, 128 `AdamW` updates, `32,768` cell calls): **`loss_before = 1.113166 -> loss_after = 0.001735`, `train_accuracy = 1.000 (256/256)`, `critical_false_accepts = 0/91`**.
   - Turn 2 (`req=18`): Re-ran the *exact same* `steps=128` config (`a2`, identical SHA `878c5651...`) instead of fitting a `steps=0` initialization baseline.
   - Turns 3–8 (`req=19..24`): Called `check` 6 consecutive times despite `error: check unavailable; learned artifacts receive a final test after submission`, never calling `submit` -> `score = 0.0`.
2. **Verifier-confirmed solutions lost to stateless 2-cycle oscillation (`boolean_repair/670612341ca76f87` & `cursor_recovery/56624b5c277b4ea5`, `qwen3-4b-instruct-2507-q8`):**
   - On both episodes, Turn 1 attempted `run_experiment` -> `error: experiment unavailable`; Turn 2 called `check` with the **exact correct answer**, receiving **`check_passed = True`**!
   - On Turn 3, because Research v2's stateless history (`task-plus-complete-current-observation`) replaced the prior error with `{check_passed: true}`, deterministic `temperature=0` decoding reverted to `run_experiment`, entering a `run_experiment -> check (True) -> run_experiment -> check (unavailable)` cycle for all 8 turns and **never calling `submit`**.
   - Burning 8 turns on every one of the first 8 episodes exhausted the entire 64-action global budget, leaving episodes 9–32 unattempted (`run_budget_exhausted`).
3. **Learner config schema hallucinations (`trm_training/3619b09983a0e7b9`, `ldt_filter/a32b45659a6b6500`, `qwen3-4b-instruct-2507-q8`):**
   - Emitted unsupported config keys (`trm_width`, `train_seed`, `learner`) -> `error: unknown or inapplicable learner config field`, then echoed the error string into `{"action":"check","answer":...}` in a 2-cycle loop.
4. **Submitting without verification (`evidence_custody/68c2e9f6eb2a9c32` & `ccb19a567c5fad67`, pinned `qwen3-4b` BF16):**
   - Unlike the 2507 instruct arm, pinned `Qwen/Qwen3-4B` (`thinking_mode=disabled`) responded to `error: experiment unavailable` by calling `submit` on Turn 2 (`20.95s` and `17.17s`), cleanly closing both episodes — but without first running `check` or chain-of-thought verification, it misclassified custody records (`score = 0.0`).

### 5.3 Serving & Compute Constraints (CPU 30s Per-Turn Cap vs. Leased GPU)
- Quantized GGUF CPU serving (`llama-server`, `--threads 8 -ngl 0`) stayed comfortably inside the 30s per-turn deadline (`12.2–16.4s` for Baby Goblin Q4_K_M; `2.2–12.5s` for Qwen3-4B-Instruct Q8_0; **0 timeouts** across both held32 runs).
- Full BF16/FP32 PyTorch CPU execution (`2–3.5 tok/s`) hit the 29s worker / 30s transport deadline on `26/32` turns for single-pass `Qwen/Qwen3-4B` BF16 and on 100% of smoke prompts for multi-pass recurrent models (`dory-4b-bf16-loop2`, `nanbeige42-3b-loop2`, `hrm-text-1b-h2-l3`, `baby-goblin-10m-v2-native`). Re-running those 5 registered bindings on an uncontested 16 GiB+ GPU requires zero code changes (`local_worker.py --device cuda`).

---

## 6. Phase 4: Externalized Persona & Jinn/Beast Specialist Experiment

Executed via [`scripts/research_v2/jinn_beast_persona_exp.py`](../../scripts/research_v2/jinn_beast_persona_exp.py) over the pinned private archive (`MoralityLabAI/jinn-or-beast-paper@01512c40`, `report_sha256` `2590bfd156d3ebc2db4c62576ef3deca4694246fbae1321220629fd2d00c0585`). Strict split separation verified: **1,440 train / 432 validation / 432 test roots** (`0` cross-split overlaps, `0` overlap with prospective `heldout-panel-20261126`, all `4,608` source labels independently recomputed).

### 6.1 Unseen-Root Policy Head Performance (432 Sealed Test Roots, 864 Frame Targets)

| External Head Architecture | Frame | Params | Guarded Acc | Raw Acc | Macro F1 | Guarded Crit FA | Raw Crit FA | Opposite-Frame Invariance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| External LDT Head (union tags) | Jinn | 75 | 96.76% | 96.30% | 0.9676 | **0 / 141** | 2 / 141 | 97.69% |
| External LDT Head (union tags) | Beast | 75 | 98.38% | 97.69% | 0.9838 | **0 / 144** | 3 / 144 | 95.14% |
| **External LDT + TRM Routing + 3-pass RLT** | **Jinn** | 198 | **98.38%** | 98.15% | **0.9838** | **0 / 141** | 1 / 141 | **100.00%** |
| **External LDT + TRM Routing + 3-pass RLT** | **Beast** | 198 | **99.31%** | 98.61% | **0.9931** | **0 / 144** | 3 / 144 | **100.00%** |

*Training progression (train-only 1,440 roots):* Standard LDT converged from loss `0.925 -> 0.177` (Jinn) / `0.922 -> 0.166` (Beast); adding 3-pass damped RLT correction (`60` epochs, 123 params) further reduced training loss to `0.089` (Jinn) / `0.071` (Beast) while TRM frame-routing structurally guaranteed **100.0% opposite-frame invariance** on both frames.

### 6.2 Multi-Owner Temporal Revision, Revocation & Intervention Comparison

#### A. Full 432 Unseen-Root Intervention Suite (`at_revision=5`, model intervention/reset at rev 3, shuffled delivery, cross-owner + character + fiction injections at rev 5, future rev 6 leakage)

| Condition / Arm (432 unseen roots) | Pref Fidelity | Temporal Rev Acc | Cross-Owner Leakage | Revocation Handling (216 cases) | Unseen-Root Policy Acc | Critical-Policy False Accepts (155 crit) |
|---|---:|---:|---:|---:|---:|---:|
| Base in-context (post-revision reset) | 7.41% | 7.41% | 58.33% (252) | 100.0% | 52.31% | **28.39% (44/155)** |
| Base in-context (full window, no critical veto) | 100.0% | 100.0% | 0.0% | 100.0% | 96.76% | 2.58% (4/155) |
| **External LDT persona state** | **100.0%** | **100.0%** | **0.0%** | **100.0%** | **97.69%** | **0.00% (0/155)** |
| **External LDT + TRM routing + RLT correction** | **100.0%** | **100.0%** | **0.0%** | **100.0%** | **99.31%** | **0.00% (0/155)** |
| Ablated external: **no owner binding** | 0.00% | 0.00% | **100.0% (432)** | 100.0% | 32.64% | **36.77% (57/155)** |
| Ablated external: **no temporal binding** | 3.94% | 9.26% | 0.0% | **47.22%** | 51.62% | **26.45% (41/155)** |
| Ablated external: **no owner / temporal / scope / revoke** | 1.62% | 4.17% | 49.77% (215) | **0.00%** | 45.83% | **30.32% (47/155)** |

#### B. Empirical Live-Checkpoint In-Context Probes (16 Unseen Jinn/Beast Test Episodes)

| Live Checkpoint Arm (16 unseen test episodes) | Parse/Transport Errors | Mean Latency | Pref Fidelity | Temporal Rev Acc | Cross-Owner Leakage | Revocation Handling | Routed Unseen Policy Acc | Critical FA (4 crit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-4B-Instruct-2507-Q8_0` (in-context, **full window**) | **0 / 16** | 6.38s | **6.25%** | 18.75% | 0.0% | **0.00%** | 50.00% | 0 / 4 |
| `Qwen3-4B-Instruct-2507-Q8_0` (in-context, **post-reset**) | **0 / 16** | 5.62s | **18.75%** | 25.00% | 0.0% | **12.50%** | 43.75% | **1 / 4 (25.0%)** |
| `Baby Goblin 10M v2 Q4_K_M` (in-context, **full window**) | 16 / 16 | 4.30s | 0.00% | 0.00% | 0.0% | 0.00% | 0.00% | 0 / 4 |

**Key Phase 4 takeaway:** Even when a strong 4B instruct model parses 100% of events cleanly and filters cross-owner injections (`0%` leakage), it fails **93.75% of the time** to reconstruct shuffled temporal revisions and revocations in context (`0%` revocation handling on full window) and drops to `43.75%` policy accuracy with critical false accepts after a model revision reset. Externalizing preference state into an owner-, scope-, and temporally-bound store coupled with a 198-parameter TRM-routed RLT LDT head eliminates both context-reset loss and in-context temporal-ordering errors while driving critical false accepts to zero.

---

## 7. Reproducibility & Artifact Inventory

- **Extensible Mesh Registry:** [`configs/research_v2/models_mesh_20261126.json`](../../configs/research_v2/models_mesh_20261126.json)
- **Serving, Evaluation & Analysis Scripts:**
  - [`scripts/research_v2/local_worker.py`](../../scripts/research_v2/local_worker.py) (Aleph native KV-cached Direct-A, HF Transformers w/ HRM PrefixLM & Nanbeige `DynamicCache`, `llama-server` adapter)
  - [`scripts/research_v2/wsl_loopback_proxy.py`](../../scripts/research_v2/wsl_loopback_proxy.py) & [`scripts/research_v2/run_model_eval.py`](../../scripts/research_v2/run_model_eval.py)
  - [`scripts/research_v2/jinn_beast_persona_exp.py`](../../scripts/research_v2/jinn_beast_persona_exp.py)
  - [`scripts/research_v2/aggregate_benchmark_results.py`](../../scripts/research_v2/aggregate_benchmark_results.py) & [`scripts/research_v2/export_sanitized_artifacts.py`](../../scripts/research_v2/export_sanitized_artifacts.py)
  - [`scripts/research_v2/build_specialist_candidate_dataset.py`](../../scripts/research_v2/build_specialist_candidate_dataset.py)
- **Immutable Receipts & Bindings:** [`benchmark_ledger.json`](benchmark_ledger.json), [`receipts/`](receipts/), [`bindings/`](bindings/)
- **Phase 6 Curriculum Proposal & Candidate Train Dataset:** [`SPECIALIST_CURRICULUM_PROPOSAL.md`](SPECIALIST_CURRICULUM_PROPOSAL.md), [`training/research_v2_specialist/candidate_train.jsonl`](../../training/research_v2_specialist/candidate_train.jsonl), [`exclusion_manifest.json`](../../training/research_v2_specialist/exclusion_manifest.json), [`DATASET_RECEIPT.json`](../../training/research_v2_specialist/DATASET_RECEIPT.json)
