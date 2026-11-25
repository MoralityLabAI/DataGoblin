# Aleph data planning

The [19 November 2026 landscape report](../reports/2026-11-19_data_landscape.md) evaluates the owned corpus receipts and public sources for a proposed 20B-token run with a conditional 30B continuation.

| File | Contents |
| --- | --- |
| [public_pretraining_catalog.json](public_pretraining_catalog.json) | 24 public pretraining and solution sources, including access, ancestry and license notes. |
| [formal_game_catalog.json](formal_game_catalog.json) | 17 formal-math/game datasets, repositories and generator families. |
| [owned_corpus_catalog.json](owned_corpus_catalog.json) | Owned corpus receipts, pinned Lean source inventories and the local checked Campsite repair pack. |
| [aleph_20b_30b_mix.json](aleph_20b_30b_mix.json) | Proposed quotas, repeat limits, fallback rules, objectives and accounting requirements. |
| [aleph_20b_30b_estimates.json](../estimates/aleph_20b_30b_estimates.json) | Reproducible analytical compute cases for 10B, 20B and 30B tokens. |

## Meaning of the records

These files are planning metadata. `training_ready` remains false. Public source sizes use the publishers' counting methods; owned receipts also use several different tokenizers and serialization formats. Aleph's tokenizer revision and accepted unique counts remain unresolved.

Source cards were read on the audit date. Pin data-file revisions at intake and retain original document, problem and episode IDs. The corpus export should contain achieved quotas after filtering, repeat caps and shortfall substitution. Keep the desired quotas beside those achieved values.

The proposed static-root cap is four presentations total across all views of a root. Up to four distinct verified views can share that allowance. Preserve original split custody during extraction, grouping all descendants of a held-out root with that root.

## Objective integration

Reference text and source code need a declared language-modeling objective. Checked tasks use response supervision with claim-specific receipts. The proposed pretraining adapter must represent provenance and split eligibility separately from truth or outcome verification. This planning update leaves the current [`data_etiquette.yaml`](../../configs/data_etiquette.yaml) policy unchanged.

Research predictions and model critiques remain attributed proposals. A proof checker, executed test suite or environment replay can support a narrower outcome claim. Admit each record with the claim type and receipt that establish its allowed supervision.

The RLT Campsite module pilot has 511 distinct, verifier-checked repair examples from 256 train puzzles. It compares direct feedback, answer-state RLT and matched damping on shared targets. Count the repair examples under the existing procedural-games lane once admitted. The local train pack has a content-hashed manifest. Full-scale admission needs a source-rights review and accepted-token counts with the frozen Aleph tokenizer. [Build and replay the training pack](../../docs/rlt_hermes_repair.md#training-pack)

## Frontier teacher-trace curriculum candidates (2026-11-24)

Status: discovery queue only; `training_ready=false`. These links are candidate sources for continued pretraining and post-training of Aleph. Community dataset names are uploader claims, not authenticated teacher identities. Availability, revisions, sizes, origin, licenses and upstream provider permissions need independent review before admission.

| Candidate source | Proposed use |
| --- | --- |
| [MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x) | Long reasoning trajectories; extract bounded decisions and subgoals. |
| [greghavens/fable-5-coding-and-debugging-traces](https://huggingface.co/datasets/greghavens/fable-5-coding-and-debugging-traces) | Debugging, failed attempts, tool feedback and repair. |
| [Crownelius/Complete-FABLE.5-traces-2M](https://huggingface.co/datasets/Crownelius/Complete-FABLE.5-traces-2M) | Broad Fable/Opus-attributed candidate pool; check overlap. |
| [clzoro/Claude-Distills](https://huggingface.co/datasets/clzoro/Claude-Distills) | Opus/Sonnet-attributed reasoning candidates. |
| [RESMP-DEV/Fable-GPT-5.5-Distillation-Traces](https://huggingface.co/datasets/RESMP-DEV/Fable-GPT-5.5-Distillation-Traces) | Mixed-teacher candidate pool; audit lineage before sampling. |
| [predictive-singularity/Accretion](https://huggingface.co/datasets/predictive-singularity/Accretion) | Coding and agent-behavior candidates. |
| [AletheiaResearch/GPT-5.5-Codex](https://huggingface.co/datasets/AletheiaResearch/GPT-5.5-Codex) | Agent-session segmentation and repair. |
| [TeichAI datasets](https://huggingface.co/TeichAI/datasets) | Frontier-model-attributed reasoning collections. |
| [Manusagents mixed frontier distillation](https://huggingface.co/datasets/Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset) | Large mixed-source discovery lead; high-priority rights and provenance audit. |

Astra Pro / GPT-6.2: no authenticated public corpus identified in the initial discovery pass. Keep a separate discovery slot; do not infer Astra authorship from GPT or Codex labels. The owned Spiral Codex campaign is a separate candidate subject to authorization, provider terms, confidentiality screening and source rights.

### Open tooling and comparison sources

| Resource | Role |
| --- | --- |
| [Teich](https://github.com/TeichAI/teich) | Investigate session import, normalization and response masking. |
| [OpenThoughts](https://github.com/open-thoughts/open-thoughts) | Reproducible reasoning-data recipe and comparison corpus. |
| [NVIDIA Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | Software-repair trajectories; replay tests independently. |
| [LeanDojo](https://github.com/lean-dojo/LeanDojo) | Formal proof states and tactic extraction; evaluate newer tooling before pinning. |

### Proposed 500M-token frontier lane

Illustrative post-filter, post-tokenization targets in millions of loss-bearing tokens: Fable 150, Opus 75, GPT/Codex 50, verified software repair 75, open mathematical reasoning 75, native Lean/research traces 75. This is a separate proposed comparison lane, with no change to the existing 20B/30B quotas or strict-1B architecture. If incorporated into continued pretraining, debit exposures against the approved total and avoid double counting. Rights failures or insufficient unique capacity require a documented shortfall or separately approved substitution.

Compare raw response SFT, compressed subgoal supervision, and state-transition supervision at matched loss-bearing token and planned recurrent-cell budgets. The last view carries state, proposal, independently checked outcome (or unknown), and revised action. Keep tool/environment output as context, mask non-target spans, and preserve failed actions as attributed proposals. Teacher self-grades do not supply independent truth labels.

Admission requires pinned dataset/file revisions, origin and teacher-attribution evidence, rights review (including applicable provider terms), secret/PII screening, deduplication across aggregates, root-level split custody and the existing four-presentation cap. Retain unknown outcomes as masked labels; verify Lean with the kernel and software repair with isolated tests. Record accepted unique and loss-bearing Aleph tokens with the frozen tokenizer, actual replay costs and source-specific exclusions. CPU/RAM inventory and sampling should precede any separately authorized GPU or Silico run. No dataset downloads or training are authorized by this catalog.

## Research conjectures: pinned external Lean source (2026-11-25)

[solved-conjectures](https://github.com/1337isnot1337/solved-conjectures/tree/e1cf80b80c2fe9d22d83f1dc3673410a2ac1bf2e) is registered in the formal-game catalog under the existing `formal_lean` lane. The pinned tree has 5,882 Lean paths (272,157,737 bytes), of which 5,215 are under `lean/` (262,579,836 bytes). These are raw source counts; result directories, archived snapshots and generated certificates do not count as independent theorem tasks. The two large filename-prefix groups Loneliness and Frankl alone contribute 246,395,879 bytes under `lean/`. Stage the primary sources before considering archives, preserve upstream credits and per-file license notices, and sample generated certificates separately. The pinned Lean 4.35.0 and mathlib lock, result-level scope reviews, strict axiom audit and independent replay are prerequisites for checked supervision. Keep public conjecture families separate from reserved evaluation roots. No source payload was downloaded, no proof was independently replayed, and no Aleph token volume or quota credit is claimed. Existing owned-corpus totals and mixture allocations are unchanged.

## Generation measurement

The proposed first yield measurement is 10,000 examples across 12–20 task families. Record accepted unique Aleph tokens, loss-bearing tokens, canonical roots, replay pass rate, duplication and resource cost. This measurement has not run.

Store raw training content, generated runs and evaluation labels outside this directory. The repository retains catalogs, plans and aggregate analytical estimates.

## Model Atlas: small-model research specimens

[Model Atlas collator](../model_atlas/README.md) stages pinned fit receipts, recipes, development benchmarks, causal and mechanistic summaries for Aleph's research-learning corpus. The first catalog targets 55 distinct archived LBF fits. Generated candidate documents are discovery-only until rights, evidence, lineage and token-admission review; no additional training quota is assumed.
