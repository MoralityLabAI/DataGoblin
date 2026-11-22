# MeTTa training corpus: pinned source and future verified rollouts

This directory collects public MeTTa programs and tests useful for DataGoblin's data-curation curriculum. The initial snapshots cover atomspace updates, graph transforms, backward chaining, static-analysis failures, JSON/CSV operations, and agent memory. They are inputs to review and execution, with licenses and content identity retained.

## Structure

- `raw/<owner>--<repo>/<upstream-path>.metta`: upstream bytes, unchanged.
- `annotations/<owner>--<repo>/<upstream-path>.metta.json`: one per source, pinned commit and Git blob digest, SPDX, task family, runtime hint, observed source assertions, dependency heuristic, and custody.
- `licenses/<owner>--<repo>/`: original upstream LICENSE and, where present, NOTICE.
- `scripts/corpus.py`: deterministic offline content/provenance/metadata validator.
- `scripts/export.py`: source-only pretraining JSONL or provenance manifest JSONL.
- `scripts/fetch.py`: restore missing snapshots from immutable upstream commits with Git blob verification.
- `rollouts/`: reserved for separately captured, independently checked interpreter traces.
- `discovery_catalog.json`: further source repositories and collection gates.

## Run the offline validation

```sh
python MeTTa/training_data/scripts/corpus.py --json
python -m unittest discover -s MeTTa/training_data/tests -v
python MeTTa/training_data/scripts/export.py --mode manifest --out runs/metta/manifest.jsonl
python MeTTa/training_data/scripts/export.py --mode source --out runs/metta/source_pretraining.jsonl
```

Source export is a language-modeling candidate. Every seed annotation has `verification_status=not_run`, `rollout_status=none`, `training_status=source_only_pending_review`, and `split=unassigned`. Source assertions were found lexically; they have not been executed. Tasks with upstream imports may require extra modules and an exact interpreter build. Negative tests preserve the source author's expected rejection, without claiming an observed failure. The corpus validator checks Git blob SHA-1 for byte identity, not execution semantics.

Approval for model training requires a separate review of license scope and data quality. Construct model-specific token counts with the actual target tokenizer; the seed annotates UTF-8 bytes and line counts, leaving `model_token_count` null. Split by repository lineage and generated task root before training to prevent code/test leakage.

## Capture real rollouts

Store interpreter version, command, process exit status, stdout/stderr hashes, parsed assertions, state changes, verifier receipt, and root lineage for each run. Record successful, failed, and halted traces with the exact runtime. Only promote a trace when its verifier receipt and immutable source identity agree; generated repair counterfactuals need their own provenance. See `rollouts/README.md`. No model performance claim follows from this initial source import.

## Redistribution and safety

The license directory preserves the upstream licenses and NOTICE files. Every source stays linked to its pinned repository revision. This dataset is public-code material and must not absorb credentials or private user data. The discovery catalog explicitly holds uncertain-license projects outside this snapshot.

## Grow the collection under a budget

The collector only accepts licensed repositories already listed in `discovery_catalog.json`, pins the existing seed commit, checks upstream Git blob digests, and defaults to a dry-run inventory. It deliberately skips large files, generated directories and bulk data paths unless explicitly enabled. All new annotations remain unexecuted and unassigned.

```sh
python MeTTa/training_data/scripts/crawl_repo.py --repo zariuq/CeTTa --max-files 100 --max-total-bytes 1000000
python MeTTa/training_data/scripts/crawl_repo.py --repo zariuq/CeTTa --max-files 100 --max-total-bytes 1000000 --apply
python MeTTa/training_data/scripts/corpus.py --json
```

Review each generated annotation and the actual program before promoting source language-modeling data; evaluate license and upstream-data-specific rights separately for graph imports. This collector intentionally makes no SFT/rollout verification claim.

## Reproducible expansion and exact tokenizer count

The initial source expansion used a one-time GitHub Actions run on PR #18. The recipe is retained under `workflow_recipes/`; future intake uses `scripts/crawl_repo.py` with explicit limits. Source files remain pending independent verification.

Measure a true token count with a named tokenizer (counts are not portable across tokenizer vocabularies):

```sh
python -m pip install 'tiktoken==0.12.0'
python MeTTa/training_data/scripts/count_tokens.py --encoding o200k_base
```

The report and one-token-count-per-source manifest live in `reports/o200k_base.json` and `token_counts/o200k_base.jsonl`. The count describes raw source-language-modeling tokens, separately from generated rollout tokens; use the actual Bonsai/Qwen tokenizer for their native training budgets.

## Extract upstream author-asserted examples

Many `.metta` test files include explicit `!(assertEqual ...)`, `!(assertEqualToResult ...)`, and `!(test ...)` claims. Extract them with source line spans and parsed query/expected-expression pairs:

```sh
python MeTTa/training_data/scripts/assertions.py
```

The generated `assertion_candidates/author_assertions.jsonl` and `reports/assertions.json` retain the source blob hash, runtime hint, task family, exact assertion text, and source-asserted result. They are marked `verification_status=not_run` throughout. A future interpreter replay and independent verifier must certify any resulting SFT target; the extractor cannot serve as a rollout labeler.

## Pinned Hyperon runtime observations

The Hyperon execution pilot uses the published `hyperon==0.2.10` interpreter to run small source tests from `trueagi-io/hyperon-experimental`. Inputs remain byte-identical, and each process has a fixed timeout. The first job has read-only GitHub permissions, and execution subprocesses receive a restricted environment. Results carry runtime version, exit code, elapsed time, full-output hashes and capped output previews:

```sh
python -m pip install 'hyperon==0.2.10'
python MeTTa/training_data/scripts/run_hyperon.py --max-files 20 --timeout-sec 12
```

Evidence goes into `runtime_receipts/hyperon_0_2_10.jsonl` and `reports/hyperon_0_2_10.json`. Exit code zero means the process completed. An independent verifier must determine whether each assertion passed, and until then every recorded attempt remains ineligible for SFT/rollout training.

## Relation to the main MeTTa curation gym

The sibling `source_catalog.json`, `looped_sft.py` and `native_query_replay.py` provide a separate discovery inventory, synthetic curation correction trajectories, and independently replayed native MeTTa query fixtures. Overlapping exact Git blobs and repeated presentations must share a lineage identifier, even across the source and generated views. Source-only data in this directory remains outside verified SFT until admitted by the existing etiquette and verifier gates.

## Connect this corpus to the frozen MeTTa curation gym

The default `MeTTa/source_catalog.json` and its existing 1,960-item synthetic curriculum are unchanged. To build an opt-in combined source catalog, first validate the 602 source snapshots and then run:

```sh
python MeTTa/training_data/scripts/catalog_bridge.py
python -m MeTTa.curation_gym \
  --catalog MeTTa/training_data/exports/expanded_source_catalog.json \
  --out runs/metta/expanded_source_curation
```

Overlapping upstream file roots are coalesced only when repository, path, and Git blob match. New sources enter the bridge with `eligibility=quarantine` and `training_ready=false`; tests ensure that no new file is automatically approved. The generated fixtures have deterministic etiquette-policy supervision in development custody, distinct from native MeTTa execution and from original-language SFT. `reports/catalog_bridge.json` records counts and exact catalog hashes.

## Native assertion results from the pinned Hyperon pilot

Audit the 20 source-level Hyperon execution receipts against the source's top-level evaluation order:

```sh
python MeTTa/training_data/scripts/native_assertions.py
```

The output `native_assertion_verdicts/hyperon_0_2_10.jsonl` stores interpreter-observed assertion results and source spans. It checks the original Git blob, full stdout/stderr hashes, exact runtime version, output count alignment and `[()]` assertion result semantics. A source test may exit successfully while returning a native `Error`; unresolved evaluations remain unobserved. Native assertion passes are empirical interpreter results, while independent expected-output and training-release gates remain separate. All generated rows retain `training_ready=false`.

## Source-conditioned native query candidate packets

The `scripts/native_query_packets.py` exporter takes the completed per-assertion Hyperon observations, checks the original source span, and extracts each successful assertion into a supervised query/expected-expression **candidate**. Context is the sequential source prefix ending immediately before the target assertion, so the target answer is withheld from its input. `training_candidates/hyperon_source_query_pairs.jsonl` uses `instruction`, `input`, `output`, and source metadata for an eventual looped-model training adapter. The exceptions JSONL retains failed, unobserved, and over-budget instances.

```sh
python MeTTa/training_data/scripts/native_query_packets.py
```

All packets remain `training_ready=false` and `split=unassigned` until overlap, lineage, semantics, and data admission are reviewed. The root identity is the original source file. Runtime agreement on author assertions is evidence of interpreter behavior; independent math/reference correctness and actual model gains require their own checks.
