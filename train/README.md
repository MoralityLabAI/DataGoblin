# DataGoblin Aleph Direct-A Continuation Trainer (`train/`)

This folder holds the ready-to-run **Aleph Direct-A 10M -> 100M (Staged 10B)** continuation trainer set supporting:
- **`--device tpu`** — Google Colab **TPU `v5e-1`** (`3.26 CU/hr`, `15.75 GiB` HBM, `49.3 GB` cloud host RAM, 24 cloud CPU cores) using native Google **TorchTPU** (`torch.compile(..., backend="tpu")`) with BF16 block working weights on TPU and exact FP32 master weights, FP32 per-microbatch gradient accumulation, and FP32 AdamW moments on cloud host CPU RAM.
- **`--device xla`** — Google Cloud **TPU `v6e-1`** (`32 GB` HBM) via PyTorch/XLA PJRT.
- **`--device cuda`** — **NVIDIA A100 40GB / H100 80GB**.

## Contents (`train/aleph_continuation10b/`)

- `runner.py` — Versioned continuation trainer with fail-closed verified handoff from `MyDrive/aleph-v2-run/group_039_7fbdb1f91cf8` (step `39`, `10,000,000` positions), preserving the frozen `999,352,320`-parameter Direct-A architecture, 24 executed visits, 2,048 context, FP32 master + FP32 AdamW moments, RNG state, prior records, and absolute LR schedule (`1.25e-5` @ 10M -> `1.25e-4` @ 100M -> `2.5e-4` @ 200M..16B -> `2.5e-5` @ 20B).
- `tpu_backend.py` — Native TorchTPU (`device="tpu"`) backend, `torch.compile(backend="tpu")` forward+loss step, and 24-core cloud-host FP32 gradient accumulator + AdamW offload (`HostOffloadFP32AdamW`).
- `stream.py` — Deterministic exact-integer deficit-first multi-lane `StageMixedStream`, `collect_stage_group`, and detached `replay_stage_release` (`90,000,000 = 343 * 262,144 + 84,608` -> `344` new updates, absolute step `383`).
- `drive_sync.py` — Parallel SHA-256-verified checkpoint sync/restore publishing `LATEST.json` strictly last and retaining `>= 2` durable generations.
- `intake_stage100m.py` — Deterministic Stage-100M corpus intake, screening, deduplication, frozen SmolLM2 (`tokenizers==0.22.2`) packing, and 344-group detached replay.
- `stage_to_drive.py` — Idempotent Google Drive release uploader with SHA-256 ledger output.
- `audit_research_10b.py` — CPU-only 10B open/frontier reasoning corpus inventory, deduplication, 1,000-root streaming sample, and 300-example calibration audit.
- `colab_stage_launcher.ipynb` — Zero-secret Colab notebook preconfigured for **`v5e-1 TPU`** (`DEVICE = "tpu"`, `ALLOW_BACKEND_HANDOFF = True`).
- `stage_100m.json` — Frozen Stage-1 (`10M -> 100M`) stage contract.

Canonical receipts and readiness report live in `training/aleph_next/continuation10b/receipts/` and `training/aleph_next/continuation10b/READINESS_REPORT.md`.
