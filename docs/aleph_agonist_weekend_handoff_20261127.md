# Aleph continuation and Agonist remote deployment

Status: operator handoff; no training job launched by this document. Launch and provider expenditure require the operator's existing authorization. Keep all keys, provider responses, checkpoint paths with access credentials, private data and generated runs outside Git.

## Friday–Sunday: continuation (ALEPH-DIRECT-A)

Use the native direct-A 999,352,320-parameter checkpoint documented in `pre-trains/aleph-direct-a-10m-v2/README.md`, group_039_7fbdb1f91cf8, with original optimizer, RNG, tokenizer, and cursor. The checkpoint is not a finished language model. Recover the implementation and receipts from PR #74 (`research/aleph-recovery-sources-20261125`, `training/aleph_next/recovery/`); pin the actual commit after verification. Verify the replacement corpus archive/index hashes and checkpoint digest before loading. Do not accidentally resume from a duplicated 24-layer Llama export, or restart the learning-rate warmup.

- Use the 10B-position continuation milestone. 20B is a stretch only if the operator has explicitly approved resources. Keep the frozen direct-A architecture and original schedule. Preserve source rights, train-only lineage and the previously established mixed corpus recipe. Build or recover the full continuation corpus; the 10M replacement release alone is not the full corpus. Do not relabel repeated 10M positions as fresh unique tokens.
- Qualify the allocated runner with an actual forward/backward/optimizer step, memory telemetry, deterministic cursor transition and a restore. CUDA single-A100 is the only demonstrated hardware path in the pilot report; do not claim distributed or TPU qualification from CPU fixtures.
- Record 5-minute steady-state useful non-padding positions/s, supervised label tokens/s, wall-clock deadline and estimated reachable endpoint. The pilot achieved 4,520 positions/s. Ten billion in 62 hours requires about 44,803 positions/s (before downtime); a single pilot-speed device delivers about 1B. Use a rolling projection from actual throughput, and preserve the highest safe achievable endpoint if the requested endpoint cannot be reached. Do not exceed provider/host caps.
- Save native weights, optimizer, RNG and cursor atomically at intervals of at most 30 minutes or 100M positions, whichever comes first. Keep at least two verified recoverable checkpoints. Export milestone sampler checkpoints and a Q4_K_M GGUF through the existing parity-checked conversion path. Record hashes and measured device memory; verify actual sampling via the native model or llama.cpp completion mode, not an invented chat template.
- At the Sunday cutoff: stop cleanly, verify restart and export parity, preserve the final checkpoint, publish only aggregate non-sensitive receipts. Report actual positions, labels, unique-source counts, costs, speed, and any blocked operations. No architecture tournament this weekend.

## Monday: DG-AGN-TINKER-01

Extend `Agonist/` and the AGN-01/AGN-02 contracts, retaining all offline deterministic fixtures. Add an opt-in Tinker worker rather than altering the existing local llama.cpp teacher contract. Implement an adapter registry and a durable queue, not a second gym.

Model: `thinkingmachines/Inkling-2-Mini` (64K training option). Pin the SDK and verify the actual tokenizer/renderer, account model access, free/paid limits and pricing via the operator's account. Environment-only secrets: TINKER_API_KEY, ANTHROPIC_API_KEY. No calls by default. New paid calls require an explicit live flag and configured spend caps.

Create independent LoRA runs initially at rank 16 (proposed, adjustable) for these closed-schema tasks:
- curriculum: rank permitted candidate IDs, nominate a next batch, predict marginal benefit, abstain;
- route: choose a registered skill/adapter or STOP, with expected cost and confidence;
- diagnose: classify observable failure codes and propose permitted checks;
- repair: produce bounded executable example revisions that pass the existing independent verifier.

Start with the base-model baseline and small SFT batches drawn from admitted, rights-cleared original or Tinker-generated verifier-labeled examples. Keep split by canonical root/generator family, preserve the four-presentations-per-root cap, and do not promote labels supplied by the actor itself. Keep sampler snapshots immutable during evaluation. Save both full training state (save_state) and sampler snapshots (save_weights_for_sampler); restore optimizer with create_training_client_from_state_with_optimizer and sample each adapter via create_sampling_client(model_path=...). Gate promotion on verified development outcomes, compare to base and previous incumbent, roll back failed candidates. Keep sealed roots outside the actor and train paths. No online RL until end-to-end SFT, restore and budget gates pass.

The local controller loads the best verified Aleph export in llama.cpp with bounded context, only if its smoke results permit useful structured decisions. Otherwise continue via deterministic queue/curation policy; record the real policy identity. A local 4GB GPU is for Aleph/small controllers only; the Inkling GPU workload remains remote. Set initial RAM and VRAM caps based on measured host headroom. Run as a supervised Windows service/task; prevent sleep on AC, recover after reboot, require authenticated private remote access, provide stop/pause, and maintain a bounded local disk budget.

### Agonist episode

Observe immutable public evidence -> propose action -> select specialist -> sample -> verify independently -> revise or stop -> measure actual downstream student delta -> record. Limit repairs and token/cost per root. Store append-only local receipts with episode_id, root_id, split, state_hash, policy and adapter snapshot identity, proposal and candidate IDs, confidence, visible evidence references, action and failure code, verifier version/result, revision lineage, student metric before/after where measured, token counts, cost and timestamps. Distinguish observed outcomes from unexecuted predictions. Preserve failures, aborts, and duplicate proposals. No fabricated internal chain of thought; record public decision summaries and tool outcomes.

Fable runtime advisory lane: use Fable 5.2 xhigh for bounded consultations on an independently retained trace and observable operational problems. Keep all consult content and influenced lineages in a separate restricted local archive; exclude them from model-training inputs, targets, rewards, and data selection for potentially competing general-purpose models unless Anthropic provides permission for the intended use. Independent clean runs can supply trainable original/Tinker-generated/verifier-labeled trajectories. Do not assume filtering or paraphrasing a Claude output makes it permissible training data. Fable must not control shell, spend, corpus promotion, or hidden evaluation access. The operator owns all permissions.

### Monday acceptance
- Offline fixture tests and existing regression checks pass; no network on dry-run.
- One live base sample, one small LoRA update, sampler checkpoint creation, distinct adapter sample, optimizer-state restore and verified resume; all individually capped and explicitly authorized.
- Replay one complete Agonist episode after process termination with no duplicate charge/retraining and preserve pending/unknown requests for reconciliation rather than automatically resubmitting.
- Demonstrate remote stop/restart, hard spend ceilings, private authentication, no secrets in logs, no access to held-out roots, and a private daily digest of valid/invalid traces, per-adapter score, spend and remaining budget.
- No claim of downstream uplift without independently measured outcomes.

References: https://tinker-docs.thinkingmachines.ai/tinker/quickstart/ ; https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/ ; https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/ ; https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model
