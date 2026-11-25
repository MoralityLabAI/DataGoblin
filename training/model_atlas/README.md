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

Next intake targets: LoopedTransformers TRM/PPT/SAE experiments, DataGoblin RLT
and LDT gyms, Compactification-Theory, EvalMath and BPD. No counts are claimed
for those until actual checkpoint receipts have been reviewed.

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
