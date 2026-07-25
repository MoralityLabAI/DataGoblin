# Curation adapter verifier

The adapter verifier measures how consistently different normalization setups apply the
`curate-game-agent-data` contract to the same four-game dataset projection. It is offline,
deterministic, content-free, and separate from the architecture training gym.

## Contract

Candidates receive only field names, typed flags, visibility, schema identifiers, and hashes.
Raw prompts, responses, private state, reasoning traces, corpus values, filesystem paths, and
authoritative labels are absent. Environment-owned labels remain in a separate steward file.

The evaluator reports schema parse rate, expected parse-status accuracy, exact normalized-record accuracy, mapped-field F1,
abstention accuracy, coverage, privacy leak rate, critical false-release rate, calibration, and
per-game slices. Privacy leakage or a false `VERIFIED_TRAIN` release fails closed. A candidate
manifest is bound to the dataset hash and must declare `public_shape_only` input and no label access.

## Deterministic smoke comparison

```powershell
$env:PYTHONPATH = "src"
python -m datagoblin.adapter_verifier demo `
  --config configs/curation_adapter_verifier.yaml `
  --seed 7 `
  --out runs/adapter-verifier-demo
```

The demo compares three mocks: a strict policy adapter, a permissive map-all negative control,
and a conservative abstain control. All remain promotion-ineligible because mocks are plumbing
tests. The comparison deliberately has no universal scalar winner.

## Learned schema-drift gate

The canonical fixture is deliberately easy for the strict policy because its source keys are
already the answer vocabulary. `drift-confirm` creates a harder, leakage-safe task: one canonical
safe field at a time is replaced by a generated alias. The frozen exact-key policy remains the
baseline, while a fitted character n-gram mapper must recover fields from transformation families
that do not occur in its fitting or calibration roles.

The benchmark keeps three custody lanes:

- Public rows contain game names, field-name shapes, typed flags, opaque lineage commitments, and
  transformation-family names. They contain no labels or corpus values.
- The candidate workspace contains labels only for `train` and `calibration`.
- The private steward file contains all labels and enters only the evaluator.

Generated lineages cannot cross splits. Fitting/calibration transformation families cannot cross
into ordinary or OOD evaluation, and a fresh confirmation is invalid unless its public schema
surface hash differs from the pilot. The fixed privacy/evaluation membrane handles private,
reasoning-bearing, ambiguous, and sealed rows; only the safe-field mapper is learned.

```powershell
$env:PYTHONPATH = "src"
python -m datagoblin.adapter_verifier drift-confirm `
  --config configs/preregistrations/game_adapter_schema_drift_v1.yaml `
  --out D:\Temp\datagoblin-schema-drift-confirmation
```

Passing this gate means only that a small fitted adapter beat exact-key normalization on the
synthetic shape task with the preregistered safety rates. It advances the same recipe-shaped
examples to the optional LoRA/QLoRA/HRM-Text architecture gate; it does not release private game
records or establish that any larger architecture is better.

The recipe ordering is grounded in primary references:

- [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9) motivates testing low-rank checkpoint
  adaptation only after a learnable task and baseline have been established.
- [QLoRA](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
  motivates a memory-bounded adapter variant; quantization does not relax evidence or custody gates.
- [SetFit](https://arxiv.org/abs/2209.11055) motivates a small supervised representation baseline
  for a label-scarce text task before spending on a larger checkpoint.
- [Valentine](https://arxiv.org/abs/2010.07386) motivates explicit schema-family holdouts and
  field-mapping evaluation instead of treating exact canonical keys as a sufficient benchmark.

## Four-game adapter tournament

`tournament-confirm` applies the schema-drift gate to several adapter profiles on identical
Centauri, They Sing, Court, and Diplomacy fixtures:

- normal full-context;
- compact context under a lower parameter budget;
- coverage pressure with a zero mapping-confidence threshold; and
- the frozen exact-key baseline.

```powershell
$env:PYTHONPATH = "src"
python -m datagoblin.adapter_verifier tournament-confirm `
  --config configs/preregistrations/game_adapter_tournament_v1.yaml `
  --out D:\Temp\datagoblin-four-game-adapter-tournament
```

Safety gates are applied before mapping metrics. The report emits ordinary and OOD results per
game, pairwise Pareto dominance, and a Pareto front for each game. It deliberately emits
`winner: null`: a compact adapter can be preferable on parameter cost while a larger adapter can
be more robust on one game's OOD schemas. Qualification must repeat on a different schema surface,
but Pareto-front movement is retained as evidence rather than hidden behind an aggregate score.

This is a **data-adapter tournament**, not yet a tournament in which models play the games.
The preregistration includes the next arena contract: admitted checkpoint candidates must use
seed-matched public observations, fixed opponent pools, symmetric seat or role rotation,
environment-owned legality and outcome receipts, and per-game results before aggregates. No
checkpoint is downloaded or trained by this command.

## Evaluate a setup against a given dataset

Supply a candidate manifest and one prediction per dataset item:

```powershell
python -m datagoblin.adapter_verifier evaluate `
  --config configs/curation_adapter_verifier.yaml `
  --dataset-manifest path\to\public_dataset\dataset_manifest.json `
  --steward-labels path\to\private_steward\ground_truth.jsonl `
  --candidate-manifest path\to\candidate_manifest.json `
  --predictions path\to\predictions.jsonl `
  --out runs\adapter-evaluation
```

The steward path belongs to the evaluator, not the adapter workspace. For a real corpus, build
the public fixture from an approved, separately reviewed projection and freeze its manifest before
running any setup. Never point candidate code at the raw D: corpus or private Codex backend state.

## Content-free corpus projection

`project-corpus` streams bounded records from an accepted AICourt export and manifested Diplomacy
exports. It emits only allowlisted Sol/Terra/Luna and reasoning-effort cohort values, recognized
field-name tokens, type classes, counts, flags, and hashes. Unrecognized field names are counted,
not copied. Raw scalar values and source paths remain in the private projection receipt.

An aggregate receipt is only a provenance claim. It does not independently verify an individual
record, so clean projected rows remain `DISCOVERY_ONLY`; explicit reasoning/session/content fields
are rejected, and private-state fields are quarantined. Diplomacy rows with no bound model metadata
are reported as `unresolved|unknown` and excluded from Sol/Terra/Luna comparisons.

After freezing the projection, `mock-setups` runs the strict policy adapter and the permissive and
always-abstain controls against that exact manifest. Reports include slices by game, source model,
reasoning effort, and their joint cohort. These slices measure adapter behavior on data generated by
each cohort; they do not rank the underlying source models' gameplay quality.

The evaluator also reports hazard saturation and the number of non-abstention opportunities. A
projection where every real row must be rejected or quarantined is explicitly marked
`safety_gate_only`. Perfect accuracy on such a projection validates refusal behavior only; it cannot
demonstrate useful mapping coverage, justify comparing source-agent quality, or make an adapter
promotion-eligible.

## Safe-field derivation lane

`project-safe-fields` creates a separate immutable derivation rather than stripping or editing a raw
record in place. Inside private custody it hashes each allowlisted field value, binds the derivation
to the exact compressed source record and policy version, and writes an append-only private ledger.
An environment-owned second pass reopens every selected record and recomputes the field and record
bindings before the public dataset manifest is finalized.

Safe fields are matched by versioned per-game paths, not by leaf name. This prevents a nominally safe
name such as `target`, `phase`, or `orders` from being selected out of a private, reasoning-bearing, or
otherwise unrelated container.

Before assigning ordinary/OOD splits, the projector hashes episode or game lineage without exposing the
identifier, unions records that share lineage, exact source content, exact safe-value content, or a
high-similarity safe-value shingle fingerprint, and assigns each connected component as a whole. The
verification pass independently recomputes those fingerprints and fails if a lineage, duplicate,
near-duplicate, or split-group family appears in both splits.

Candidates still receive no values or paths: only safe field names, value type classes, cohort
metadata, flags, and derivation hashes. Faithful extraction does not verify the source model's game
claims, so real rows remain `DISCOVERY_ONLY` with `MODEL_INFERRED` authority. This lane creates
legitimate mapping opportunities for adapter evaluation without making a training-release claim.

## Blind semantic-review gate

`build-semantic-review` selects a deterministic, four-game-stratified sample from a frozen safe-field
projection. It re-verifies every selected value against the immutable derivation ledger, places raw safe
values only in two separate private reviewer workspaces, and emits a content-free public queue. Reviewers
use different blind identifiers and may write only typed scores, enumerated failure/rationale codes, and a
`HUMAN_REVIEWED` disposition. Unsupported axes must be `null`; free-text rationale and `VERIFIED_TRAIN`
are rejected.

The same pass computes private semantic fingerprints for the bounded full projection and emits only hashes,
item identifiers, and similarity scores in the near-duplicate audit. `finalize-semantic-review` validates both
blind annotation files and reports agreement by game/model/effort slice. Any disagreement requires a third
content-bound adjudication pass. Even unanimous review never authorizes training: independent or
environment-owned outcome evidence remains a separate gate.

When `finalize-semantic-review` receives both `--evidence-registry` and `--evidence-receipts`, it reopens every
receipt through the trusted store and requires an exact record-hash subject, proposer binding, passing verdict, and
the claim type `OUTCOME_VALIDITY`. Unknown verifier strings, registry drift, receipt replay against another item,
or receipts that certify only rule compliance cannot satisfy the outcome gate. Missing items are emitted as a
content-free `evidence_challenges.jsonl` queue naming the required per-game replay contract. Even complete evidence
for the bounded review sample cannot authorize a corpus-wide training release.

`verify-semantic-evidence` consumes that challenge queue in a private workspace. It reopens each exact indexed
source record, checks the safe-field derivation commitments, binds the record to an environment-owned native
episode or hash-bound provenance archive, and runs the game-specific replay declared in
`configs/evidence_methods`. The registry commits the SHA-256 of each method file, so editing a replay contract
invalidates old receipts. The verifier emits a receipt for every challenge, but only a `PASS` receipt carrying
`OUTCOME_VALIDITY` can satisfy finalization; missing historical engines and state-hash drift remain
`INCONCLUSIVE`.

The game methods deliberately differ:

- Centauri requires the selected actor/turn/phase/legal-command tuple to occur exactly once in the native
  research episode and requires the environment replay to reproduce all state commitments.
- They Sing binds both selected order lists to the native faction/turn records, reconstructs a replay-only
  configuration, and compares every completed-turn post-state hash.
- Court consumes the complete decision shard under an explicitly supplied engine snapshot whose version equals
  the canonical episode's `env_version`, then compares all turns and the complete outcome.
- Diplomacy reads only `lmvsgame.json` from the hash-bound provenance archive, independently adjudicates every
  saved phase, and recomputes the public outcome projection. Player sessions, responses, prompts, press, plans,
  and reasoning are excluded.

All paths, safe values, replay inputs, environment reports, and receipts remain under the private output root.
Only code, method specifications, and deterministic tests belong in the repository. A complete bounded review
sample still cannot authorize a corpus-wide training release.

```powershell
$env:PYTHONPATH = "src"
python -m datagoblin.adapter_verifier build-semantic-review `
  --config configs/semantic_review.yaml `
  --dataset-manifest path\to\public_dataset\dataset_manifest.json `
  --derivation-ledger path\to\private_projection\derivation_ledger.jsonl `
  --approved-root D:\approved-corpus-root `
  --out D:\Temp\semantic-review

python -m datagoblin.adapter_verifier verify-semantic-evidence `
  --challenges D:\Temp\semantic-review-final\evidence_challenges.jsonl `
  --review-map D:\Temp\semantic-review\private_steward\review_map.jsonl `
  --derivation-ledger D:\Temp\safe-field-gym\private_projection\derivation_ledger.jsonl `
  --evidence-registry configs\game_evidence_verifiers.yaml `
  --method-specs configs\evidence_methods `
  --approved-root D:\approved-aicourt-corpus `
  --approved-root D:\approved-diplomacy-corpus `
  --aicourt-corpus-root D:\approved-aicourt-corpus `
  --centauri-root C:\projects\AIpha-CentaurAI `
  --they-sing-root C:\projects\TheySing `
  --diplomacy-engine-root C:\projects\AI_Diplomacy `
  --diplomacy-corpus-root D:\approved-diplomacy-corpus `
  --court-engine-snapshot 1.2.0=D:\private-engine-snapshots\court-1.2.0\src\court\engine.mjs `
  --out D:\Temp\semantic-evidence-replay

python -m datagoblin.adapter_verifier finalize-semantic-review `
  --review-manifest D:\Temp\semantic-review\public_review\review_manifest.json `
  --review-map D:\Temp\semantic-review\private_steward\review_map.jsonl `
  --annotations D:\Temp\semantic-review\reviewer_a\annotations.jsonl `
  --annotations D:\Temp\semantic-review\reviewer_b\annotations.jsonl `
  --near-duplicate-audit D:\Temp\semantic-review\private_steward\near_duplicate_audit.json `
  --evidence-registry configs\game_evidence_verifiers.yaml `
  --evidence-receipts D:\Temp\semantic-evidence-replay\evidence_receipts.jsonl `
  --out D:\Temp\semantic-review-final
```
