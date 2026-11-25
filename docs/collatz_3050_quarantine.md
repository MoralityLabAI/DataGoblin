# COL-DG-02: quarantine native-gym roots

The [BPD-COL-02 laptop gym](https://github.com/MoralityLabAI/BPD/blob/f1d489a000406e2d8732f4ea7a39e5473392efe2/mechinterp/bpd/experiments/collatz_3050/README.md) uses exposed development parameters. Reuse the existing DataGoblin certificate generator and independent unaccelerated checker to validate each recipient/effect/sham role and record its basin. Build a conservative exclusion set containing every selected parameter and gcd-normalized parameter. This prevents future curriculum construction from accidentally training on native gym cases or scaled duplicates.

`PYTHONPATH=src python -m datagoblin.training_gym.collatz_3050_guard PATH_TO_RUN/cases_manifest.json`

Add `--check-corpus PATH_TO_EXISTING_COL_JSONL` to audit existing verified training rows; an overlap exits 2. The guard produces an audit only and admits zero records or token quota. It reads neither model predictions nor hidden activations, and does not change existing datasets. A nonoverlap result is not a training admission or a sealed-test claim.

The pinned actual four-root manifest is a fixture. Regression tests exercise replay, tampering and a scaled-root witness. This is an exact arithmetic audit, not a learned result.