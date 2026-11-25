# Model Atlas / Collator v1

Model Atlas is DataGoblin's research-specimen collection lane. The purpose is to
learn from populations of trained small models, including failures, training recipes,
benchmarks, mechanistic measurements, causal interventions, formal properties and
architecture history. It is part of the Aleph curriculum **discovery** catalog;
it does not authorize model training or source admission.

## Record notation

`MA1[family/study/architecture@sSEED;P=PARAMETERS;K=PASSES]`

The stable specimen identity is the reported checkpoint SHA-256, not an architecture
name or an evaluation arm. The eight fields of each model record are:

| Field | Meaning |
| --- | --- |
| genotype | Architecture, parameter count, pass count, pinned implementation |
| training_recipe | Registered recipe, training seed, reported fit and draw receipts |
| phenotype | Per-training-seed benchmark results, with study, policy, split and budget |
| mechanistic_profile | Link to study-scoped actual probe summaries and raw artifact URIs |
| causal_evidence | Link to study-scoped paired comparisons, shams and failed screens |
| formal_properties | Link to theorem/certificate receipts and scope limitations |
| lineage | Architecture predecessor and population group; unknown weight parent stays null |
| checkpoint | Reported digest and source directory; bytes are never downloaded or loaded |

The study record stores aggregate controls, measured probe summaries, budget curves,
negative screens, receipts and interpretation notes once. It is not a model.
Calibration rates are not counted as independent outcomes across training seeds.
An inference-time sham or guarded/cached evaluation of an existing checkpoint
becomes a phenotype observation, not a new model. Raw tensors and weights remain
in the upstream archive or an operator-approved local store.

## First population

The pinned catalog covers LBF-14, 15, 16, 17 and the **open calibration** phase of
LBF-18. The expected census is 4 + 12 + 12 + 24 + 3 = **55 distinct fits**, and
five study records. LBF-18 reuses LBF-17 checkpoints in its evaluation variants.
LBF-17 historical false claims are retained with the referee correction, not silently
promoted to proven facts. Study comparisons reflect the original accounting conventions.

The initial catalog intentionally excludes the LBF-18 sealed summary, all suite
files, raw episode rows, hidden evaluation seeds, full tensors, proof witnesses and
weights. The LBF family shares a population group; for research-prediction evaluation
hold out entire architecture families/study lineages, not random rows or seeds.

## Run

From DataGoblin root after installing the package:

```sh
python -m datagoblin.model_atlas \
  --sources training/model_atlas/sources.json \
  --cache runs/model-atlas/source-cache \
  --out runs/model-atlas/first-import \
  --fetch-metadata
```

Without the explicit fetch flag, place each exact pinned JSON file in the cache as
`<source_id>.json` and run offline. The Git blob hash is checked before parsing.
Fetching is bounded to configured immutable public JSON metadata paths, and
redirects are rejected. No upstream code executes. The output directory must be fresh.

Outputs: `models.jsonl`, `studies.jsonl`, `candidate_documents.jsonl`,
`inventory.json` with deterministic content hashes and census. Generated outputs
stay outside tracked source; the dedicated CI workflow stages an artifact for review.

## Add the next population

Start from the source catalog contract: pinned repo/commit/blob, explicitly
open-development summary, study/family grouping, independent root count, budget,
registered recipe and reviewed interpretation notes. For the common fit manifest
adapter, emit `schema`, `sealed_outcomes_opened: false`, `fits` with
`arch` (or `arm`), `seed`, `parameters`, `passes`, and checkpoint
SHA-256; optionally `seed_rates`, `seed_penalized_work`, aggregate
`rates`, `costs`, `paired`, `screens`, `budget_curves`,
`fresco_probe_summary` and `training`. Add the reviewed source to the catalog.
Different manifest layouts require a narrow adapter plus tests, not a recursive
scrape of all JSON files in a repo.

## Repository queue, extracts and the coverage ledger

`repositories.json` is the ordered intake queue. Each entry carries a status (`extracted`, `partial`, `queued`,
`no_artifacts`, `inaccessible`, `excluded`), the pinned extract files (commit SHA-1 and file SHA-256), and the open
items a reader needs: unresolved provenance, missing artifacts and notes. `extracts/` holds one deterministic JSON per
pinned commit. Collation is offline: it verifies each extract digest, merges them with the curated LBF sources and
writes the corpus, the population inventory and the ledger.

```sh
# 1. Extract metadata from a pinned commit (reads Git objects only; no checkout, no weights, no upstream code).
git clone --bare --filter=blob:none https://github.com/OWNER/NAME NAME.git
python -m datagoblin.model_atlas_extract --git-dir NAME.git --ref <branch-or-sha>   --repository OWNER/NAME --out training/model_atlas/extracts/NAME.<sha12>.json
# 2. Pin it in repositories.json, then collate (add --sources/--cache to include the curated LBF fit manifests).
python -m datagoblin.model_atlas_corpus --queue training/model_atlas/repositories.json   --sources training/model_atlas/sources.json --cache runs/model-atlas/source-cache --out runs/model-atlas/corpus
```

What the extractor keeps: tree entries (path, size, Git blob SHA-1), the names of weight files, Git blob SHA-1 of
every JSON receipt, and for dicts holding a checkpoint digest: numeric and Boolean fields, short identifier labels
(architecture, arm, optimizer), public base-model ids, pass/fail verdict flags, and the run's recipe container
(`args`, `config`, `hparams`). It never keeps free text, weight bytes, per-root arrays or any field whose key names a
witness, label, token, hidden or sealed quantity. Paths with a custody token (`sealed`, `holdout`, `hidden`,
`private`, `ordinary`, `ood`, `witness`, `labels`, ...) are not parsed at all and are counted in the ledger.
A `results_file` named by a run manifest contributes its per-loop summary metrics and multiple-choice denominators.
JSONL files (episode rows and traces) are counted but not collected.

Identity and deduplication:

- A model is identified by the digest the source reports (`sha256:<hex>`); a tracked weight file that no receipt
  names is identified by its Git blob (`gitblob:<sha1>`), taken from the tree entry without reading the bytes.
- One digest reported by several receipts or studies is one model with several phenotype rows, never several models.
- `checkpoint.artifact_kind` separates `final_or_primary` checkpoints from `intermediate_snapshot`, `sae_dictionary`
  and `transcoder` artifacts. Design cells and primary-model counts ignore seeds and exclude the other kinds.
- A digest that is only ever referenced (a teacher or base checkpoint) has no model record; it is listed under
  referenced-only identities in the inventory and the ledger. A parent edge is set only when exactly one co-located
  parent-role digest exists; otherwise `weight_parent` stays null and the candidates are listed.
- `population_group` is a heuristic that drops version, recovery and ordinal suffixes from the study directory name.
  Hold out whole groups when evaluating prediction, and treat the heuristic as a starting point for review.

Fields the generic scan cannot supply stay null or `unknown`: implementation URI, recurrence topology, metric
definitions, causal effect sizes and formal-verification status. `mechanistic_profile`, `causal_evidence` and
`formal_properties` carry receipt-level tags (name-derived, not verified) and links to those receipts. Every
phenotype row is `source_reported` with `independent_verification: false`; negative-result and correction documents
are linked from their study records and never filtered out.

`COVERAGE_LEDGER.md` and `corpus_report.json` are the committed outputs of the latest collation. The JSONL corpus
stays untracked under `runs/`; `corpus_report.json` records its SHA-256 digests so a regeneration can be compared.

## Corpus admission

All outputs are discovery records with `training_ready=false`. A Git blob match
establishes source identity, not correctness, license or independent verification.
Before generating loss-bearing Aleph text or SFT, require source-rights review,
claim-specific evidence, secret/PII screening, family/root-level split separation,
cross-source deduplication and token/repeat accounting under the existing
[pretraining data policy](../data/README.md). Do not infer authority from a model's
self-report. Group raw mechanistic data under the same custody as its source roots.
Do not turn a retrospective outcome into a prediction input. The candidate
documents serialize model and study facts; future prediction examples need
an explicit input/target temporal firewall and separately frozen evaluations.
