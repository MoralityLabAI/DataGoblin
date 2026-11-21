# First real pretrained-model RLT STOP corpus: 7M TRM, 2026-11-21

Source: [BPD learned controller PR #140](https://github.com/MoralityLabAI/BPD/pull/140).

The native pretrained TRM campaign actually trained a six-feature logistic
stopping model and evaluated it on independent Sudoku source roots.
It used the SHA-pinned gaoxin492 7M Attention-TRM checkpoint, not Inkling 2 Mini.
Its genuine native CPU run solved 47/48 exposed development puzzles with fixed
depth 2 and with learned exit. The learned exit used 75 rather than 96 recurrent
steps, and 17.284104 rather than 22.288943 measured native CPU seconds.

The independent
[BPD audit and export Actions #44912385166](https://github.com/MoralityLabAI/BPD/actions/runs/44912385166)
replayed the exact Sudoku verifier, controller scores, calibration threshold,
and random stopping control from the complete source receipts. It produced
an artifact, bpd-learned-rlt-exit-audited-dg-train, with exactly:

- 96 TRAIN-only verified STOP/CONTINUE SFT examples;
- 96 TRAIN-only second-pass outcome examples;
- 45 TRAIN-only verified one-step repair preference pairs;
- a SHA-256-bound manifest and an independent audit JSON.

The source manifest SHA is
dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1.
The exact pretrained model SHA is
932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.

## Ingest in DataGoblin

Download the artifact ZIP from the linked GitHub Actions run, then run:

    python -m datagoblin.training_gym.rlt_gym.pretrained_exit_ingest \
      --source path/to/bpd-learned-rlt-exit-audited-dg-train.zip \
      --out runs/datagoblin/trm_exit_train_20261121

The CLI also accepts a directory containing the BPD four-file export
manifest, controller_sft.jsonl, outcomes.jsonl and repairs.jsonl.
ZIP archives are read by named member, without extraction or path traversal.

The importer checks file digests, model identity, training-eligibility,
each independent root and task digest, first-step Sudoku validity,
correct STOP decisions, second-step success, and chosen/rejected
repair pairs. It rejects duplicate roots, missing pairs, nonfinite
logit-derived features, changed schemas and any calibration or
development examples in the training corpus.

Output:

- controller_train.jsonl: 96 small-model training examples with task
  question, currently proposed answer, six observable logit features,
  verified STOP/CONTINUE target and model/source provenance.
- repair_preferences.jsonl: 45 exact-verifier-backed positive repair
  pairs from first/second native recurrent steps.
- manifest.json: source and output SHA-256 hashes and counts.

## Training and interpretation

These are genuine **training-root** observations. It is appropriate
to use them for a DataGoblin STOP-controller SFT pilot. The original
48 calibration and 48 development worlds and their labels were held
outside the exported training examples and remain evaluation-only.

The features are specific to the pretrained Sudoku model's 9-digit
readout. A DataGoblin recurrent controller on Lean, MCP, research
planning or hypergame tasks will need new corresponding task-specific
state features and verifier outputs, collected with the same source-root
separation. Keep the native TRM's model-state recurrent steps distinct
from DataGoblin RLT visits and Inkling's architectural repeated blocks.

No cross-task performance transfer is claimed. The parent Inkling
teacher gym still awaits a memory-suitable real Inkling checkpoint run
and a native cross-loop reader intervention.

## Checks

    python -m pytest -q tests/unit/test_rlt_gym_pretrained_exit_ingest.py
    ruff check src/datagoblin/training_gym/rlt_gym/pretrained_exit_ingest.py \
      tests/unit/test_rlt_gym_pretrained_exit_ingest.py

The private DataGoblin repository has a persistent Actions runner
assignment failure: jobs complete before runner setup, with empty
step lists and no usable logs. Source and tests are therefore
committed for local validation; BPD's independent source/corpus
audit and export checks have already passed on a public runner.
