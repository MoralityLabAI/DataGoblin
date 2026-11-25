# Direct A: single-TPU 10M qualification pilot

Keep RECIPE.json and the 10M data index frozen. This is a bounded engineering pilot, not a production launch. The original CUDA runner is unchanged. The new colab10m_xla.py uses PyTorch/XLA PJRT, one device, static [1,2048] inputs, explicit BF16 dense projections with differentiable casts from FP32 masters, FP32 residuals/norm/loss, all-position fixed-shape loss, per-visit rematerialization, shared block identities, and FP32 AdamW moments. It executes the native 24-visit direct schedule. Attention uses explicit GQA expansion and masked matmul/softmax rather than a specialized TPU attention kernel. Compilation, actual peak HBM and throughput must be measured.

## Zero-credit gate

From repo root:
    python training/aleph_next/candidate01/colab10m_xla.py --self-test --device cpu
    python training/aleph_next/candidate01/colab10m_xla.py --self-test --device cpu --bf16

The committed public CPU job at 15b92a8 passed both plus the original CUDA runner's CPU fixture: https://github.com/MoralityLabAI/DataGoblin/actions/runs/45643966527 . Four supervised labels across three packs (including an entirely masked pack); native FP32 loss 4.1913512945, port FP32 4.1913518906, BF16 4.1873683929; FP32 maximum gradient error 6.55651e-7 and relative L2 error 3.73105e-7; BF16 relative L2 error .00880476. Document isolation checks logits and document-specific gradients. All-masked loss/gradient zero. The FP32 update agrees with native AdamW to the declared tolerances. Both fixture restarts match the uninterrupted second update to 1e-6. No full-size updates. The follow-up public XLA CPU job exercises the actual compiler on the small BF16 fixture; its results must be checked separately.

Locate the operator's train-only Aleph-10M-train.zip (original Windows path C:\Users\patri\Documents\Codex\2026-11-24\they\outputs\Aleph-10M-train.zip). Check SHA-256 8042336235ba39ae1786416d55d7fdf3f641bd2ed43c609b2ef11e59c3ea25b1 and run its verify_bundle.py on fresh extraction. Do this before provisioning. Index SHA-256 be25522ded491e0c0c3f81d87d786034b46733e1ccbe2b9cea2dafc47b48abf4. The runner checks each emitted update group's digest, masks, positions and valid-label count before using it. No private held data is required.

## Target qualification

Use an authenticated Colab TPU v6e-1 session with one addressable TPU, >=28 GB reported memory, enough host RAM to construct the 4 GB CPU model, and checkpoint storage. Inspect the runtime's installed PyTorch, torch_xla and libtpu first. Use a supported matching package set, e.g. torch==2.9.0 and torch_xla[tpu]==2.9.0 if appropriate for the actual runtime. Do not set XLA_USE_BF16 or XLA_DOWNCAST_BF16; these can compromise the FP32 master/optimizer contract. No distributed/SPMD mode. Record versions and actual hardware.

    export PJRT_DEVICE=TPU
    python training/aleph_next/candidate01/colab10m_xla.py --self-test --device xla

The fixture is a gate, including predeclared BF16 tolerances. A failure stops the pilot; diagnose, retain evidence and fix the implementation before proceeding. It is not evidence of full-size TPU memory fit.

After confirming the compute-unit price and authorization, set BASE to the extracted bundle's mix/ directory, and OUT to a fresh, operator-controlled persistent directory with sufficient capacity and verified filesystem semantics. The first command performs exactly one complete frozen group (262,144 positions); checkpoint only after a finite completed AdamW update. Choose the seconds cap from actual billing allowance. Example, conditional on remaining budget:

    python -u training/aleph_next/candidate01/colab10m_xla.py --device xla --execute --base "$BASE" --out "$OUT" --max-groups 1 --wall-seconds 1200 --checkpoint-reserve-seconds 180

Inspect TPU_FIXTURE.json, PROGRESS.json, CHECKPOINT_001.json, XLA_METRICS.txt and LATEST.json. Verify all checkpoint hashes and restart on identical topology. One real update is the first engineering milestone. The 999M model has approx 16.0 GB master+gradient+moments before activations, dense BF16 casts, attention and logits, compiler workspaces; static geometry alone is not memory qualification. Use XLA metrics to diagnose compilation, fallbacks and recompile counts. HBM reports are runtime snapshots/peak when exposed.

A bounded continuation, conditional on measured time, cost, parity and disk availability:

    python -u training/aleph_next/candidate01/colab10m_xla.py --device xla --execute --base "$BASE" --out "$OUT" --resume --max-groups 39 --wall-seconds 1800 --checkpoint-reserve-seconds 180

The example wall caps are not forecasts or permission to spend. Set them to actual remaining authorized runtime. The process deadline is cooperative (a compilation, execution or checkpoint can overrun it); an external controller must enforce the compute-unit/time budget and disconnect/delete the runtime, including on failures. Budget provisioning, setup, compile, validation, checkpoint copies and disconnect. Do not start the 10B run. The script does not access Google auth tokens or start/stop Colab itself.

## Checkpoint semantics

Each saved generation streams one named parameter and its FP32 moments at a time to CPU, with model/optimizer identities, source hashes, logical step, RNG state and exact lane cursor. Verify hashes before publishing LATEST.json. Restore checks all hashes before copying tensors. Approx 12 GB per full generation, with space for the old generation and temporary new generation. No old checkpoint is deleted automatically. For a target of 39 from scratch reserve at least 39 GB free for the three scheduled saves, plus copies/backup and the bundle. Drive mounts must be checked for rename/durability guarantees: use local POSIX staging and a fully verified persistent copy with its pointer published last where necessary. Check persistence before disconnecting.

The soft deadline reserves time for checkpointing and discards incomplete gradient accumulation. It uses the cursor captured at the most recent complete optimizer step. PROGRESS.json distinguishes durable_group from completed in-memory work. On a crash recover the published pointer. A complete fixture restart is demonstrated on CPU; full-size device restart parity and actual external-storage failure behavior remain unverified. The port deliberately rejects cross-backend/topology resumes and does not qualify distributed training.

## Current limits

Single-device engineering port, no TPU result yet; no accelerator credits spent by the assistant. CPU fixture tests exercise 8-position miniature tensors, not a real 2048-position training batch or the local operator ZIP. The TPU implementation's optimizer uses an explicit FP32 AdamW formula matched against native AdamW in the CPU fixture. Full-size TPU optimizer and accumulation numerics, hardware fit, full-size restart, and throughput are separate qualification gates. Global label normalization is for one device; distributed reduction is not implemented. No trained checkpoint is claimed until a real run produces one.
