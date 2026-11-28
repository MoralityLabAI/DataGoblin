# Aleph Direct-A 10M -> 100M (Staged 10B) Continuation Readiness Report

**Date:** 2026-11-28  
**Branch:** `feat/aleph-10b-continuation-prep-20261128`  
**Prior Qualified Runner Commit:** `b078ab75274a2f95449b1daf99b80529049663e7` (PR #74, preserved untouched)  
**Prior Durable Native Checkpoint (Read-Only):** `MyDrive/aleph-v2-run/group_039_7fbdb1f91cf8` (step `39`, `10,000,000` positions, manifest SHA-256 `aca6d5c46301b6bedda375ad3f597dcff29aa4bd7fd1ed0c97677e678a9858ab`)

## 1. Stage-by-Stage GO / HOLD Verdict

| Stage | Cumulative Positions | New Positions | Corpus Status | Runner / Tests | Compute Gate | Verdict |
|---|---:|---:|---|---|---|---|
| Pilot (completed) | 10M | 10M | Frozen | Qualified A100 | Completed | **DONE (Immutable)** |
| **Stage 1 (`stage_100m`)** | **100M** | **90M** | **Admitted, replayed (344 groups), staged on Drive** | **CPU + live TPU `v5e-1` fixture & benchmark PASS (`3,275 pos/s`, `24.88 CU`)** | **Authorized on 1× `TPU v5e-1` (`3.26 CU/hr`); ready to launch upon Drive OAuth consent** | **TPU-QUALIFIED / READY TO RUN** |
| Stage 2 (`500M`) | 500M | 400M | Sources audited; intake pending Stage-1 checkpoint | Ready (`train/aleph_continuation10b/runner.py` scales to 10B) | ~110.6 CU on 1× TPU `v5e-1` (~201 CU on A100) | **HOLD** |
| Stages 3–6 (`2B / 5B / 8B / 10B`) | 10B | 9.5B | Open & frontier reasoning audited (`1,000` roots / `300` calibration); rights review ongoing | Ready; YaRN longer contexts require separate qualification | ~2,626 CU on TPU `v5e-1` (~5,010 CU on A100) | **HOLD** |

## 2. Empirical TPU `v5e-1` Qualification, TPU `v6e-1` Probe & Colab Account Audit

- **Live TPU `v5e-1` Benchmark Receipt:** `training/aleph_next/continuation10b/receipts/tpu_v5e1_benchmark_receipt.json`
- **Google Cloud `TPU v5e-1` (`3.26 CU/hr`, `15.75 GiB` usable HBM, `49.3 GB` host RAM, `24 vCPU`, `PyTorch 2.14.1` + native TorchTPU) — HARDWARE QUALIFIED:**
  - Exact Direct-A hardware fixture passed: `gradient_relative_l2 = 0.00935` (`<= 0.12`), `isolated_document_gradient = true`, `all_masked_zero_gradient = true`, `unequal_label_counts = true`, `fixture_restart = true`.
  - Per-block `torch.compile(..., backend="tpu")` + exact on-TPU FP32 gradient accumulator + 1×/group 24-thread cloud-host FP32 AdamW (`train/aleph_continuation10b/tpu_backend.py`):
    - Warmup compile: **`13.04s`**; static XLA scratchpad: **`601 MB`** (down from `9.05 GB` monolithic); total allocated HBM: **`6.84 GB / 16.91 GB`** (`10.07 GB` free headroom, zero OOM).
    - Exact FP32 gradient parity: `grad_norm = 2.0601` (matches monolithic compiled to 4 decimal places).
    - Sustained end-to-end throughput: **`3,275.4 pos/s`** (`0.541s/mb`, `80.03s` per `262,144`-position group including D2H/AdamW/H2D).
    - Projected full Stage 1 (`90M` tokens, `10M -> 100M`): **`7.63 hours`**, **`24.88 CU`** (vs A100 `45.13 CU` — **44.9% cheaper**).
- **TPU `v6e-1` Probe Receipt:** `training/aleph_next/continuation10b/receipts/tpu_v6e1_probe_receipt.json`
  - Both authenticated `colab v0.8.0` (`--tpu V6E1`) and live Colab Web UI (`RuntimeService/CreateAssignment`) stalled indefinitely server-side (`0.00 CU` spent). **TPU `v6e-1` remains UNQUALIFIED.**
- **Colab Account State:** `971.01 CU` available; two pre-existing unrelated CPU sessions (`shz-j1`, `shz-j2`, `0.34 CU/hr` combined) untouched.
- **Published Trainer Set:** Mirrored cleanly into `train/aleph_continuation10b/` (with `stage1_tpu_driver.py`, `eval_dev_panel.py`, `runner.py`, `tpu_backend.py`, `colab_stage_launcher.ipynb`).

## 3. Workstream A — Immutable 90M Stage-1 Corpus Release & Drive Ledger

- **Receipt:** `training/aleph_next/continuation10b/receipts/stage_100m_release_receipt.json`
- **Tokenizer Freeze:** Exact SmolLM2 `tokenizer.json` SHA-256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`, `vocab=49152`, `tokenizers==0.22.2`, `encode_special_tokens=True` (literal control-token protection verified).
- **Custody & Screening:**
  - `100%` of `math.libretexts.org` and `/book/math-*` excluded (`2,887` rows rejected in `libretexts_filtered`, `79` in `finemath-4plus`); zero pilot records recycled.
  - Cross-source exact & near deduplication, secret/API-key/PII-email/SSN/benchmark-contamination screening enforced per document with full per-source rejection ledgers.
  - Deterministic root-family split (`5%` source families strictly separated into `dev_panel/`, `95%` eligible for train-only `stage_100m/`); `max_presentations_per_root = 1` (`<= 4`); zero wraparound.
  - `HuggingFaceTB/stack-edu` recorded as deferred metadata-only (Software Heritage IDs, no executable source bytes); `common-pile/comma_v0.1_training_dataset` skipped in favor of `allenai/peS2o` v2 for explicit per-document provenance (`id`/`source`/`created`).

| Lane | Pinned Source(s) | Target Positions | Replayed Train Positions | Replayed Supervised Labels | Train Records / Families | Held-Out Dev Positions |
|---|---|---:|---:|---:|---:|---:|
| `educational_reference` | `common-pile/libretexts_filtered@204d01e40a` (non-math CC-BY/CC-BY-SA/CC0/PD) | 40,000,000 | **40,000,000** | 39,960,945 | 19,534 / 373 | 255,436 |
| `scientific_papers` | `allenai/peS2o@636a503e44` (`data/v2/train-00000-of-00020.json.gz`, LFS-ETag verified stream, ODC-BY) | 10,000,000 | **10,000,000** | 9,959,006 | 36,128 / 36,125 | 250,098 |
| `code_tests` | `OpenCoder-LLM/opc-annealing-corpus@078e4bc1d3` (`algorithmic_corpus/{1,10}.jsonl`, ODC-BY) | 20,000,000 | **20,000,000** | 19,893,477 | 96,796 / 54,039 | 250,138 |
| `informal_math` | `HuggingFaceTB/finemath@e92b25a616` (`finemath-4plus/train-00000-of-00064.parquet`, ODC-BY) | 20,000,000 | **20,000,000** | 19,977,554 | 12,689 / 4,282 | 250,418 |
| **Total** | | **90,000,000** | **90,000,000** | **89,790,982** | **165,147 / 94,819** | **1,006,090** |

- **Detached 344-Group Replay Verification (`REPLAY.json`):**
  - `343` full groups (`262,144` pos) + `1` final partial group (`84,608` pos) = `344` updates (`39 + 344 = 383` absolute steps at `100,000,000` positions).
  - `43,947` packs, `3,456` padding positions (`99.996%` packing efficiency).
  - Mid-stage restart parity after group `172` (`saved_cursor_sha256 = 7ee04b15a294d8f3d5e1b4824e662dd8070ce2609b3f75677d621bf73e1b11c3`, `terminal_cursor_sha256 = f766df5c2100aec23dddfec04bedff7e44b67b3ce7ea1128902f94235efc397a`): **`restart_parity_verified = true`**.
- **Durable Google Drive Staging (No Unauthorized HF Dataset Created):**
  - Root Folder (`MyDrive/aleph-10b-continuation-v1`): `https://drive.google.com/drive/folders/1Q_PNU-iHQuQ42y82dsjHQ8ikfhrU57s5`
  - Immutable Train-Only Release (`stage_100m`, 40 files, `1,032,108,851` bytes): `https://drive.google.com/drive/folders/1CWzzeKqPhqSzNdmJRPIsZYui0LJfPhrz`
    - `index.json` SHA-256: `194a8b0febce37de7497c34b9d20c8f1e10879a1065fe297a8da5d4c607ae232` (`https://drive.google.com/file/d/1HCRdsfxTGdpvZq7OXMlj143Iy7vuwAia/view`)
    - `REPLAY.json` SHA-256: `41a18f99e0287057f253fc2b1e599fe538cebb396d35ce59d4e23f4e07fba7cd` (`https://drive.google.com/file/d/18CUdIaJzZbsbs4ZmagzjD7DjJt3x45A3/view`)
    - `MEMBERS.json` SHA-256: `52c9e18a238520b7a78a7e44d48e3ed050dc52096360f787fa9d206e3d96b960`
    - Full Drive ID ledger: `training/aleph_next/continuation10b/receipts/drive_stage_100m_ledger.json`
  - Strictly Disjoint Held-Out Dev Panel (`dev_panel`, 22 files, `20,587,281` bytes): `https://drive.google.com/drive/folders/1cHuNhzGbFj63Bn2kQQUQ7ELY4jJCdzz-`
    - `dev_index.json` SHA-256: `131cb6c80fae1a7063c58fa72a693fdbb92dc3b4dbd5c3084cbfd3bab3bd84bf` (`https://drive.google.com/file/d/1DhLoBbhUsVf8K_fUuZwpKBrN-rG829J-/view`)
    - Full Drive ID ledger: `training/aleph_next/continuation10b/receipts/drive_dev_panel_ledger.json`
  - Colab Launcher Notebook: `https://colab.research.google.com/drive/1o1KuKO7s3H60Ho1Aj3fjR9HyYpJ26kTt`

## 4. Workstream B — Native Training Migration & Verification

- **Imported Pinned Lineage Modules (PR #74 `b078ab75`):** All 5 SHA-256 hashes match `MyDrive/aleph-v2-run/IDENTITY.json` byte-for-byte (`recipe`, `colab10m_xla.py`, `colab10m.py`, `gym.py`, `packing.py`). Original `verify_bundle.py` and `colab10m_xla.py` remain untouched.
- **Versioned Continuation Implementation (`training/aleph_next/continuation10b/`):**
  - `stream.py`: exact integer-arithmetic deficit-first multi-lane `StageMixedStream`, `collect_stage_group`, `replay_stage_release`, fail-closed exhaustion/wraparound/cross-lane-root checks.
  - `runner.py`: strict one-time verified handoff (`verify_and_handoff_prior_checkpoint`) preserving FP32 master weights, FP32 AdamW moments, RNG state, prior 39 records, and absolute LR clock (`1.25e-5` @ 10M -> `1.25e-4` @ 100M -> `2.5e-4` @ 200M..16B -> `2.5e-5` @ 20B); supports `--device cpu|cuda|xla`.
  - `drive_sync.py`: parallel SHA-256-verified checkpoint sync/restore supporting both legacy (`state_sha256`) and continuation schemas, publishing `LATEST.json` strictly last and retaining `>= 2` durable generations.
  - `colab_stage_launcher.ipynb`: zero-secret operator-parameterized Colab launcher with Drive read/write probe, 1-group qualification mode, capped continuation mode, and optional auto-disconnect.
- **Measured CPU Self-Tests & Unit Tests (`receipts/cpu_migration_selftest.json`):**
  - `runner.py --self-test --device cpu` (FP32): `PASS` (`grad_rel_l2 = 4.72e-7 < 1e-4`, `isolated_document_gradient = true`, `all_masked_zero_gradient = true`, `handoff_and_resume_parity_passed = true`).
  - `runner.py --self-test --device cpu --bf16` (BF16): `PASS` (`grad_rel_l2 = 8.80e-3 < 0.12`, `handoff_and_resume_parity_passed = true`).
  - `pytest tests/unit/test_aleph_continuation10b.py`: `4/4 PASS` (including fail-closed negative tests for bad step, bad position offset, bad manifest pin, altered continuation identity, corrupted shard, tampered cursor, cross-lane duplicate root, and split mismatch).
  - `ruff check`: `0 errors` across all new continuation modules and tests.

## 5. Workstream D — Parallel 10B Open & Frontier Reasoning Audit (CPU Only)

- **Receipt:** `training/aleph_next/continuation10b/receipts/research_10b_intake_audit.json`
- **Zero Paid Teacher Calls (`paid_teacher_calls = 0`).**
- **Inventory & Rights Findings:**
  - `14` linked HF datasets + `25` `TeichAI` org datasets + `9` formal/bespoke GitHub repositories inventoried.
  - `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces`: **UNAVAILABLE** (`HTTP 401` / removed or private).
  - `Crownelius/Complete-FABLE.5-traces-2M` (`license: null`, raw SDK CLI event logs) and `AletheiaResearch/GPT-5.5-Codex` (`license: null`, GPT-5 Codex harness logs — explicitly **not** Astra): **quarantined for unknown rights**.
  - `greghavens/fable-5-coding-and-debugging-traces` and `clzoro/Claude-Distills`: HF `datasets-server` returned `HTTP 500` (flagged for direct file review before admission).
  - `nvidia/OpenMathReasoning`, `nvidia/OpenScienceReasoning-2`, `open-thoughts/OpenThoughts3-1.2M`, `nvidia/OpenCodeReasoning-2`: live, large-scale open supply prioritized for 500M+ stages (separating card license vs teacher ToS review).
  - **PR #121 (`DG-Research-20M`, `b74b4c0d512e`):** inventoried as a separate bespoke native research intake workstream (preserving custody exclusion of reference-only provider responses; not mixed into Stage-100M).
- **1,000-Root Streaming Sample & 300-Example Independent Calibration (SmolLM2 `0.22.2`):**
  - Sampled **`1,000` distinct roots** (`17,521,983` total tokens).
  - Deduplication removed `82` intra-dataset duplicate roots in `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`, `10` in `OpenThoughts3-1.2M`, and `3` cross-source overlaps between `Accretion` and `Fable-5.1-Max`.
  - Primary `>= 70` research-relevance filter accepted `91` roots (`1,718,365` total tokens, `1,647,905` loss-bearing tokens); `30` roots (`452,676` tokens) passed both `>= 70` relevance and strict `permissive_cleared` rights; `141` roots flagged `unknown_outcome_masked`.
  - Independent 300-example structural calibration: **`precision = 97.14%`** (`TP=34, FP=1, FN=227, TN=38`, `recall = 13.03%`), confirming the `>= 70` rubric is ultra-high-precision and suitable for filtering multi-million-row open reasoning pools (or tunable downward on a fresh calibration split if higher recall is desired for 2B+ tranches).

## 6. Next Authorized Operator Action

To transition **Stage 1 (`10M -> 100M`)** from **HOLD** to **GO**, approve an explicit Colab A100 40GB spend cap:
1. **Qualification Cap (~5–10 CU):** Run `MODE = "qualify_1_group"` (1 update group, native checkpoint save, Drive hash verify, and matched restart parity check) via `colab_stage_launcher.ipynb`.
2. **Full Stage-100M Cap (~65–75 CU including save/restore/eval overhead):** Run capped resumable sessions (`344` total new updates -> absolute step `383` at `100,000,000` positions), evaluate on the disjoint `dev_panel/`, and publish the immutable 100M checkpoint before planning Stage 2 (`500M`).
