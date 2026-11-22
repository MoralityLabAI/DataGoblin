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
python Metta/training_data/scripts/corpus.py --json
python -m unittest discover -s Metta/training_data/tests -v
python Metta/training_data/scripts/export.py --mode manifest --out runs/metta/manifest.jsonl
python Metta/training_data/scripts/export.py --mode source --out runs/metta/source_pretraining.jsonl
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
python Metta/training_data/scripts/crawl_repo.py --repo zariuq/CeTTa --max-files 100 --max-total-bytes 1000000
python Metta/training_data/scripts/crawl_repo.py --repo zariuq/CeTTa --max-files 100 --max-total-bytes 1000000 --apply
python Metta/training_data/scripts/corpus.py --json
```

Review each generated annotation and the actual program before promoting source language-modeling data; evaluate license and upstream-data-specific rights separately for graph imports. This collector intentionally makes no SFT/rollout verification claim.

## Reproducible expansion and exact tokenizer count

The seed expansion workflow `.github/workflows/metta-expand-seed.yml` collects up to a bounded set of pinned examples and tests from eight approved repositories and commits them with their license, source digest and custody labels. It runs one time when its workflow definition is added to the seed PR branch. Upstream sources remain unexecuted.

Measure a true token count with a named tokenizer (counts are not portable across tokenizer vocabularies):

```sh
python -m pip install 'tiktoken==0.12.0'
python Metta/training_data/scripts/count_tokens.py --encoding o200k_base
```

The report and one-token-count-per-source manifest live in `reports/o200k_base.json` and `token_counts/o200k_base.jsonl`. The count describes raw source-language-modeling tokens, separately from generated rollout tokens; use the actual Bonsai/Qwen tokenizer for their native training budgets.

## Extract upstream author-asserted examples

Many `.metta` test files include explicit `!(assertEqual ...)`, `!(assertEqualToResult ...)`, and `!(test ...)` claims. Extract them with source line spans and parsed query/expected-expression pairs:

```sh
python Metta/training_data/scripts/assertions.py
```

The generated `assertion_candidates/author_assertions.jsonl` and `reports/assertions.json` retain the source blob hash, runtime hint, task family, exact assertion text, and source-asserted result. They are marked `verification_status=not_run` throughout. A future interpreter replay and independent verifier must certify any resulting SFT target; the extractor cannot serve as a rollout labeler.
