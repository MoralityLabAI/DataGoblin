# Neon execution handoff: Aleph direct-A, 10M to 100M, then staged 10B

Execute the first continuation preparation on the local machine and Colab CPU runtime. Deliver working code, admitted shards and qualification receipts; do not merely write another design. Use a new Git branch from current master, preserve concurrent work, and import the required runner modules from the pinned PR 74 head with original lineage. Inspect local working trees and open PRs before touching them. Never rewrite a previous release or checkpoint.

## Inputs

- Git: https://github.com/MoralityLabAI/DataGoblin
- Active 10B plan: https://github.com/MoralityLabAI/DataGoblin/blob/master/training/data/aleph_10b_mix.json
- Stage contract: training/aleph_next/continuation10b/stage_100m.json
- Runbook: training/reports/2026-11-28_aleph_10b_colab_handoff.md
- Pinned prior code: https://github.com/MoralityLabAI/DataGoblin/tree/b078ab75274a2f95449b1daf99b80529049663e7/training/aleph_next
- Pilot receipt: pre-trains/aleph-direct-a-10m-v2/README.md
- Native checkpoint: MyDrive/aleph-v2-run/group_039_7fbdb1f91cf8
- Reported private HF model repo: https://huggingface.co/AlephFunk/aleph-direct-a-10m-v2
- Proposed private stage dataset repo (not created or verified): https://huggingface.co/datasets/AlephFunk/aleph-10b-stage-100m-v1

Use authenticated local Git/HF/Drive sessions or credentials already configured by the operator; do not display tokens or copy them into notebook cells, Git, manifests, or logs. Do not upload raw data, private sessions or model weights without permission and source rights. Use Drive for durable staging by default.

## A. CPU corpus preparation

Build an immutable 90M-new-position release: educational 40M, scientific papers 10M, code/tests 20M, mathematical derivations 20M. Use the source queue in stage_100m.json. Resolve exact dataset revisions/files and rights before intake; start with the known LibreTexts candidate and sample the other sources by streaming. Stack-Edu metadata is not executable source bytes. Keep each source's original license and attribution; review content and use rights, including for teacher-generated derivatives. Reject unknown rights, secrets, PII, contamination, and unusable rows. Split by source family/root, deduplicate across current sources and the pilot, preserve no more than four static-root presentations across all views, and keep held-out data separate. No wraparound on exhaustion. Use the exact frozen SmolLM2 tokenizer and document-isolated 2048-token packer, record positions and loss-bearing labels separately. Preserve masked prompts and tool observations; verifier-specific labels require independent checks. Implement bounded deterministic downloads, restartable CPU ingestion, source-level rejection counts, and a train-only release with complete immutable hash and replay manifests. Report shortfalls; do not fabricate quota or replay the pilot. If 90M cannot be admitted, publish a smaller explicitly authorized stage contract before training. Stage data before accelerator provisioning.

## B. Native training migration

Start from the original 39-step / 10M-position checkpoint, including FP32 master weights, AdamW moments, RNG, absolute schedule clock and records. Do not load the inference GGUF or expanded Llama export as a trainer checkpoint. Original recipe has 999,352,320 unique weights, 24 executed blocks, 2048 context, vocab 49152. Preserve direct A and BF16 compute. The actual pinned schedule warms up over 200M positions to 2.5e-4, stays there through 16B, and decays to 2.5e-5 at 20B. At 100M LR is 1.25e-4; 10B is before terminal decay. No schedule change without a versioned decision.

Create a new versioned stage runner with CPU fixture and CUDA A100 modes, using the qualified original math and group semantics. Old verify_bundle and colab10m_xla are intentionally hard-pinned to 10M/39 groups. Retain their strict checks. Implement a one-time verified handoff binding original manifest and old identity, old index/cursor, tokenizer/recipe/code, absolute optimizer step and new release hash. Retain all old records and create a new release cursor without resetting optimizer or RNG. Subsequent restores must validate the new identity exactly. Verify 343 full groups of 262144 positions and an 84608-position final group, 344 new optimizer steps, total absolute step 383 and 100M absolute positions. Test exact schedule, no future leakage, document isolation, all-masked handling, gradient denominator, disjoint splits, bad hashes, wrong optimizer state, incorrect offsets and interrupted/resumed equality. Run tests on CPU and produce a reproducible notebook or launcher that reads paths and a capped resource configuration from operator inputs. Record every failure honestly.

## C. Colab qualification and continuation

Before requesting a paid accelerator, verify actual balance/rate, current authorized session and total cap, available durable Drive storage, release hashes and CPU tests. If no explicit cap exists, deliver the CPU-ready packet and request a cap. The previous A100 qualification was about 4520 positions/sec at 8.16 CU/hour; 90M adds roughly 5.53 compute hours and 45.1 CU before overhead. Do not retry failed billed allocations automatically. TPU remains unqualified. Start with one A100 40GB after approval; check durable read/write/hash and checkpoint publication before training. Run a single bounded update, save and verify a native checkpoint, run matched restart parity, then proceed in capped resumable sessions only on GO. Meter wall time, accelerator units, allocations, evaluation, restore and save costs. Keep last good generation plus staging; publish pointer last and disconnect before cap. Evaluate a fresh, fixed development panel independently without exposing held-out data to training. Save the 100M checkpoint and plan the next 500M tranche; do not launch an uncapped 10B job. Use later 1B/2B/5B/8B/10B boundaries as appropriate and preserve absolute accounting. Larger context/YaRN is a separate qualified branch.

## D. Research corpus parallel intake (CPU only)

Inventory all frontier sources already linked in the active 10B plan and the public catalog; use Fable full release rather than adding overlapping lite counts, and deduplicate mixed aggregates against original NVIDIA/OpenThoughts sources. Teacher names are uploader attributions until verified. Do not infer Astra identity. Prioritize math discovery, experiment design, ML architectures/mechinterp, scientific code, statistical analysis, proof repair, evidence revision and literature synthesis. Separate relevance from rights and truth. Use permitted teacher outputs, not leaked private traces; retain bounded research episodes with independently verifiable outcomes or unknown masks. Sample 1000 distinct roots, calibrate relevance on 300 independently reviewed examples, record unique accepted token yield, and stage admitted sources for later milestones. Prefer open reasoning sources for early supply; no paid teacher calls without a separately approved cap. Keep bespoke native research material staged as it becomes admissible.

## Deliverables

1. PR containing the real continuation runner, corpus intake/packer, tests, notebook or launch commands, and stage manifest/receipts.
2. Exact immutable Git commit and release URIs, corpus hashes and achieved counts by lane; if explicitly approved and redistribution rights permit, private HF release with dataset commit hash, otherwise durable Drive URI/path and hash ledger.
3. CPU tests and first hardware parity receipts or explicit blockers.
4. GO/HOLD, remaining actual CU budget, expected next checkpoint and current durable native checkpoint path. Do not claim 10B corpus supply or successful training without receipts.

