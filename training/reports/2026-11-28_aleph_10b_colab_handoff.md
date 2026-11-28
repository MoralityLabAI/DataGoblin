# Aleph direct-A: 10B Colab continuation handoff

**Active target, 2026-11-28:** 10,000,000,000 cumulative non-padding positions. The [10B curriculum](../data/aleph_10b_mix.json) supersedes the older 20B/30B recommendation for the initial run. The older manifests remain historical scaling alternatives. Status: HOLD on accelerator launch until the new runner and first admitted shard release pass their gates.

## Preserve the completed pilot

The [10M receipt](../../pre-trains/aleph-direct-a-10m-v2/README.md) reports a complete native direct-A run and full-size restart parity on an A100. Restore `MyDrive/aleph-v2-run/group_039_7fbdb1f91cf8` using its original manifest; copy the immutable checkpoint to a new continuation workspace and check every file hash before loading. Keep the original directory untouched. Do not use the GGUF or Llama-expanded export for training continuation. Verify actual unique trainable parameters (999,352,320), the SmolLM2 tokenizer SHA-256 (9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c), tied weights, direct feedback, 24 executed blocks, 2,048-token context, explicit BF16 compute and FP32 master weights/moments. Retain warmup and the absolute position clock (10M already processed, 39 optimizer steps including one partial group). The frozen recipe (`candidate01/RECIPE.json`, PR 74 head `b078ab75274a2f95449b1daf99b80529049663e7`) sets warmup to **200M positions**, peak LR 2.5e-4, stable through 16B and cosine decay to 2.5e-5 at 20B. Thus LR at 10M is 1.25e-5 and at 100M is 1.25e-4. Preserve that schedule by default; 10B is a checkpoint before terminal decay. A revised decay policy requires an explicitly versioned schedule and a separate decision, without restarting warmup or optimizer moments. The previous 20M warmup statement in this runbook and the pilot summary was a transcription error.

## First corpus release: 10M to 100M

The first new release is 90M processed positions: 40M educational reference, 10M scientific papers, 20M code/tests, 20M mathematical derivations. Start with rights-reviewed public source shards while bespoke corpus work continues. A manifest must pin source revision and file checksums, rights/attribution, tokenizer hash, each root and source family, split, lane, objective, actual packed non-padding and loss-bearing counts, masks, and repeated presentations. Globally deduplicate with the pilot and reserved families; exclude unknown-rights data, secrets and evaluation labels. Pretraining content is source-attributed rather than a verified fact claim. No static root appears more than four times across all views. Keep private/raw payloads and sealed roots out of Git. Stream and preprocess on CPU or an unbilled runtime; stage immutable shards on durable storage before GPU allocation. Audit the 90M packed positions and replay the first/last groups on CPU. All 90M positions must be newly processed after the existing checkpoint; no replay of the 10M prefix.

At subsequent checkpoints add OpenMathReasoning, OpenScienceReasoning-2, OpenThoughts3, OpenCodeReasoning-2 and independently admitted Fable/Opus/GPT-attributed research episodes. The JSON gives exact stage budgets through 500M, 2B, 5B, 8B and 10B. Defer unavailable data or publish an explicit revised, reconciled mixture. Do not count duplicate aggregates, inflate a lane by repeating the 10M corpus, or substitute a new teacher for an attributed source. Research relevance and rights are separate admission gates. Preserve distinct hypothesis/action/outcome/revision supervision, with unknown grades masked and independent proof/test/replay evidence as appropriate. YaRN/longer contexts are a separate branch after matched 2K retention and position/mask/recurrence/throughput checks; 2K remains the qualified initial context.

## Runner migration (blocking)

The source runner is on [PR 74](https://github.com/MoralityLabAI/DataGoblin/pull/74), branch `research/aleph-recovery-sources-20261125`. Relevant files are `training/aleph_next/recovery/train_replacement_xla.py`, `training/aleph_next/recovery/verify_bundle.py`, and `training/aleph_next/candidate01/colab10m_xla.py`. The existing path enforces a 10M/39-group release, a fixed corpus/index hash, and exact checkpoint identity; it cannot simply accept a larger --max-groups. Keep its qualification and old release unchanged.

Implement a versioned continuation entrypoint and detached replay for the new shard index. Use a one-time handoff receipt binding the old checkpoint manifest hash, recipe/model/tokenizer/code hashes, prior absolute optimizer step and position, old cursor and new release/index hash. Load and verify weights, FP32 moments, RNG, optimizer step and prior records against the original identity, then deliberately create a new cursor at the next release boundary. Preserve the old records, immutable checkpoint and cumulative position ledger; record the identity transition separately. Subsequent restarts must match the new identity exactly. Never disable the old identity check or rewrite the old checkpoint. Keep existing 262,144-position group size, allow the boundary remainder (90M = 343 full groups + 84,608 positions; 344 new updates, absolute optimizer step 383 at 100M, assuming all groups complete), and weight gradients by valid labels across microbatches. Keep causal document isolation, all-masked pack handling, and exact group counts. Compare a short uninterrupted run to interrupt/resume on CPU fixtures and on one A100 before extending the cap. Verify that corpus mismatch, altered hashes, wrong optimizer step and stale cursor fail closed. Evaluate frozen public-dev panels separately before and after the handoff; no test labels enter the trainer.

## Colab session sequence

1. CPU: prepare/train-only shard manifest, rights review, pack/replay, checkpoint hash validation, migration tests and budget receipt. Do not provision a GPU for downloads or data assembly.
2. A100 40 GB: mount Drive and test read/write/hash/atomic checkpoint publication before constructing the model. Restore using the verified native checkpoint (the receipt reports a 11.99 GB full checkpoint). Keep at least two checkpoint generations plus staging space; allow generous reserve for saves, copies and hash verification. Confirm free HBM and actual billing rate.
3. Run one bounded update group at the original context and precision. Confirm finite loss/gradient, expected position and label counts, and absolute LR; make the new native checkpoint durable. Run matched restart parity on the next groups. Stop for a GO/HOLD review if any fail.
4. Continue in explicitly capped sessions toward 100M. Save at least at session end and on a measured interval; sync to Drive, verify hashes, publish pointer last, keep the last good generation. Leave time for the save and disconnect the accelerator after the cap. Do not automatically retry allocations that bill without delivering a runtime.
5. Before 500M and later milestones review measured end-to-end positions/sec, total compute units, accepted remaining supply, fresh held-out loss and verified task measures. At 10B preserve native training and inference exports and optionally design the next run.

## Budget estimate at the qualified A100 rate

The pilot reported ~4,520 positions/sec and 8.16 compute units/hour. From the durable 10M checkpoint, arithmetic-only training estimates (excluding restore, evaluation, saving, idle and allocation overhead) are:

| Target | New positions | GPU hours | Colab units |
| --- | ---: | ---: | ---: |
| 100M | 90M | 5.53 | 45.1 |
| 500M | 490M | 30.11 | 245.7 |
| 2B | 1.99B | 122.30 | 998.0 |
| 10B | 9.99B | 613.94 | 5,010 |

Last recorded balance was 974.92 units; verify current balance and rate. At the historical rate that supports at most ~1.94B additional positions before overhead. Reserve headroom and set a separately approved per-session spend cap; the initial 100M estimate excludes overhead. A faster accelerator/backend, improved throughput or additional allocation is required to complete 10B on this budget. TPU remains unqualified. No launch, spend, fresh data admission or performance claim is made by this handoff.

## Launch receipt

Record the new release and checkpoint hashes, frozen tokenizer/recipe, achieved per-lane processed and loss-token counts, repeat and split audit, backend and resource caps, paired restart evidence, fixed dev scoring snapshot, durable checkpoint location, current account balance and GO/HOLD. The target manifest is a plan; it does not assert availability of 10B admitted tokens or qualify an unimplemented migration.

Use the [Neon execution packet](../aleph_next/continuation10b/NEON.md) and [stage contract](../aleph_next/continuation10b/stage_100m.json) for implementation and intake.
