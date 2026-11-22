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
