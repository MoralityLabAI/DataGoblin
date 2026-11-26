# DataGoblin DG-Gym v0.7 Phase 1: PR #79 Replay, Integration Review & Local Hardware Benchmark

Status: Public development only; local hardware, no paid compute.
Reviewed PR: `MoralityLabAI/DataGoblin#79` (`research/dg-gym-resource-calculus-v02-20261126`, commit `0cb1271395a39d91a967d15cab49cb69698fcf68`).

## 1. PR #79 Integration & CI Review

- **Remote CI (`0cb1271395a39d91a967d15cab49cb69698fcf68`):** All 5 GitHub checks pass (`test`, `CodeQL`, `Analyze (python)`, `Analyze (javascript-typescript)`, `Analyze (actions)`). Mergeable state: `clean`.
- **Local Lint & Unit Tests:**
  - `ruff check .` passed with 0 errors across the entire repository.
  - All 13 PR #79 unit tests (`tests/test_dg_gym_frontier.py`, `tests/test_dg_gym_resource_plan.py`, `tests/test_dg_gym_steering_feedback.py`) pass in `0.49s` under `PYTHONPATH=src`.
  - Full repository `pytest` passed 629+ tests on Windows with 1 pre-existing Windows-only CRLF failure unrelated to PR #79 (`tests/unit/test_rlt_gym_pretrained_exit_ingest.py::test_cross_split_leakage_rejected`, where `Path.write_text` writes `\r\n` while `hashlib.sha256(text.encode())` hashes `\n`).
- **Windows Portability Note:** `research/dg_gym/mod7_feedback_gain.py` imports Unix-only `resource` for `ru_maxrss`. The frozen script is left untouched to preserve its pinned SHA-256 (`98443dae180323b2cdfb8362a9663509a3ebf583cfa26e7e274c3e169e34a7d3`); local Windows replay used a non-invasive `psutil` shim without modifying any tracked source bytes.
- **Cross-Repo Context Inspected:**
  - `DataGoblin` sibling branches: `research/dg-mms-mod7-20261126` (`src/datagoblin/training_gym/mms.py`, `research/dg_gym/mod7_mms.py`) and `ops/gym-ram-watchdog-20261126` (`tools/gym_ram_watchdog.py`).
  - `LoopedTransformers`: `research_gym/interp/models.py` (`LoopedDecoderLM` `LOOP-T`/`FF-U`) and `research_gym/interp/tasks.py` (trace-annotated `pointer_chase`, `gated_pointer_chase`, `masked_pointer_chase`, `reachability`, `sudoku4`, `modular_recurrence`).
  - `BPD` (`MoralityLabAI/BPD` & `param-decomp`): rank-1 / low-rank parameter decompositions and component attribution interfaces.
  - `HybridTRMLDT` (`ldt_mesh` typed mesh units & J-objective router) and `EvalMath` (`col-certified-basin-benchmark`, `vad01`).
- **Integration Disposition:** PR #79 is clean and ready for integration; all negative experimental results (`MOD7-CPU`, `MOD7-BIJ`, `MOD7-FB-01`, `MOD7-FB-02`) are preserved unmodified.

## 2. MOD7-FB-02 Replay Verification

- **Pinned Source Verification (exact match):**
  - `research/dg_gym/mod7_feedback_gain.py`: Git blob `109d92fb1fe46970d231360b55fd8a7b3f9a8b45`, SHA-256 `98443dae180323b2cdfb8362a9663509a3ebf583cfa26e7e274c3e169e34a7d3`.
  - `research/dg_gym/mod7_feedback.py`: Git blob `eca7044bc902665cf859da417b9ab31580e980d7`, SHA-256 `5b580eef574fac8aec4aeda32489beca3f70e64c3b19cdb2113c77d975bac304`.
- **Replay Runtime:** PyTorch `2.11.0+cu128` (CPU deterministic, 1 thread), Windows 11, process peak working set `624,456 KiB` (`~639.4 MB`).
- **Determinism:** Two independent local replays matched all 15 checkpoint payloads byte-for-byte (`EXACT_15_CHECKPOINT_MATCH: True`).
- **Outcome & Accounting Verification (vs `MOD7_FB_02_RESULTS.md`):**

| Arm | Length-512 Mass (Replay) | Length-512 Mass (PR #79) | Max Row TV | Updates | Credits | Median Wall (s) | Median CPU (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 0.40527133 | 0.40527133 | 0.00385624 | 600 | 1,010,112 | 2.5310 | 2.0938 |
| `static_shadow` | 0.40534581 | 0.40534581 | 0.00374726 | 600 | 1,010,112 | 2.0620 | 2.0312 |
| `shuffled_shadow` | 0.40535516 | 0.40535515 | 0.00371102 | 600 | 1,010,112 | 2.0620 | 2.0156 |
| `static_reinvest` | 0.44900452 | 0.44900452 | 0.00318287 | 644 | 1,008,576 | 2.1410 | 2.1562 |
| `task_reinvest` | 0.57967980 | 0.57967980 | 0.00203615 | 989 | 1,009,728 | 2.7340 | 2.3906 |

- Paired gains vs controls (`adaptive - control`): `-0.0000744863` (static shadow), `-0.0000838299` (shuffled shadow), `-0.0437331930` (static reinvest), `-0.1744084685` (task reinvest). Registered screen: **FAILS** (confirmed).
- Full replay receipt with all 15 checkpoint SHA-256 hashes: `research/dg_gym/receipts/mod7_fb_02_replay_receipt.json`.

## 3. Local Hardware Benchmark

Host: **AMD Ryzen AI 9 365** (10 cores / 20 logical threads), **47.12 GiB RAM**, **NVIDIA GeForce RTX 5080 Laptop GPU** (17,094.5 MB VRAM, CUDA 12.8/12.9, PyTorch `2.11.0+cu128`).

| Workload | Device / Threads | Throughput | Wall (s) | CPU (s) | Peak Memory |
| --- | --- | --- | ---: | ---: | --- |
| Dense FP32 MatMul `1024x1024` (50 iters) | CPU 1 thread | 66.91 GFLOP/s | 1.6048 | 1.5469 | — |
| Dense FP32 MatMul `1024x1024` (50 iters) | CPU 4 threads | 169.38 GFLOP/s | 0.6339 | 2.3438 | — |
| Dense FP32 MatMul `1024x1024` (50 iters) | CPU 10 threads | 374.93 GFLOP/s | 0.2864 | 2.5781 | — |
| Dense FP32 MatMul `2048x2048` (200 iters) | RTX 5080 CUDA | 14.418 TFLOP/s | 0.2383 | — | 92.41 MB VRAM |
| Dense BF16 MatMul `2048x2048` (400 iters) | RTX 5080 CUDA | 50.830 TFLOP/s | 0.1352 | — | 92.41 MB VRAM |
| `GRUCell(128,128)` `B=96, L=8` (100 steps) | CPU 1 thread | 78.3 steps/s (60,098 visits/s) | 1.2779 | 1.2188 | — |
| `GRUCell(128,128)` `B=96, L=8` (100 steps) | CPU 4 threads | 48.8 steps/s (37,510 visits/s) | 2.0474 | 5.8281 | — |
| `GRUCell(128,128)` `B=96, L=8` (100 steps) | RTX 5080 CUDA | 111.7 steps/s (85,762 visits/s) | 0.8955 | 0.8594 | 1,259.93 MB RSS |

**Hardware Scheduling Takeaway:** For tiny recurrent widths (`d <= 64`, sequential Python loops), single-thread CPU outperforms 4-thread CPU (avoiding thread-sync overhead) and avoids CUDA kernel-launch latency; at `d = 128` (`GRUCell`), CUDA wins by `1.43x`; at dense `2048x2048` matmuls, RTX 5080 reaches `14.4 TFLOP/s` (FP32) and `50.8 TFLOP/s` (BF16).
