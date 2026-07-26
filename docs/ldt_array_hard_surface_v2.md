# LDT array hard-surface v2

This experiment tests whether small pairwise LDT arrays can preserve curation
etiquette when evidence is close to the equivalence tolerance, interval-valued,
missing, or globally nondominated. It is a synthetic meta-selection task. The
sorters remain recommendation-only and cannot promote game records.

## Frozen surface

[`configs/preregistrations/ldt_array_hard_surface_v2.yaml`](../configs/preregistrations/ldt_array_hard_surface_v2.yaml)
was frozen before generation. The generator emits:

- 16 development and 12 held-out arrays with five candidates each;
- all 20 directed comparisons within every array, for 320 development and 240
  held-out pairs;
- disjoint split lineages and context categories;
- `LEFT`, `RIGHT`, `TRADEOFF`, and `ABSTAIN` pair targets;
- `SELECTED`, `DOMINATED`, `UNRESOLVED`, and `UNSAFE` array targets; and
- public held-out inputs with pair and array targets in separate
  environment-owned steward files.

`TRADEOFF` means that fully observed quality vectors are genuinely
nondominating. `ABSTAIN` means that missing or interval evidence prevents a
conservative conclusion. The array reducer applies status precedence in this
order: unsafe, dominated, unresolved, selected.

The raw encoder receives interval centers, widths, missingness indicators,
safety, parameter buckets, and training-fit categories. The typed encoder adds
explicit safety, robust-dominance, equivalence, efficiency, insufficiency, and
tradeoff predicates. Held-out-only context categories therefore map to zero
rather than extending the fitted vocabulary.

## Candidate array

The gym compares three non-trainable controls with five learned candidates:

- raw and typed joint linear heads;
- a typed width-four, one-pass latent head; and
- width-four weight-tied heads with two or four recurrent loops.

All learned heads jointly predict preference and rationale. Three deterministic
seeds use leave-one-development-pressure-lineage-out evaluation. Held-out pair
and array targets are unavailable during fitting. The frozen policy control is
an environment-rule ceiling, not learned architecture evidence.

## Reproduce

Keep generated datasets, checkpoints, predictions, and steward reports outside
the repository:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_ldt_array_hard_surface.py `
  --config configs/preregistrations/ldt_array_hard_surface_v2.yaml `
  --out D:\Temp\datagoblin-ldt-array-hard-surface-v2

.\scripts\run_ldt_array_hard_surface_capped_wsl.ps1 `
  -DatasetManifest D:\Temp\datagoblin-ldt-array-hard-surface-v2\public_dataset\dataset_manifest.json `
  -OutDir D:\Temp\datagoblin-ldt-array-hard-surface-v2-run
```

After the capped receipt passes, perform the single steward-owned reveal:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_ldt_array_hard_surface.py `
  --dataset-manifest D:\Temp\datagoblin-ldt-array-hard-surface-v2\public_dataset\dataset_manifest.json `
  --predictions D:\Temp\datagoblin-ldt-array-hard-surface-v2-run\capped_run\heldout_pair_predictions.jsonl `
  --trainer-summary D:\Temp\datagoblin-ldt-array-hard-surface-v2-run\capped_run\trainer_summary.json `
  --pair-labels D:\Temp\datagoblin-ldt-array-hard-surface-v2\private_steward\heldout_pair_labels.jsonl `
  --array-labels D:\Temp\datagoblin-ldt-array-hard-surface-v2\private_steward\heldout_array_labels.jsonl `
  --out D:\Temp\datagoblin-ldt-array-hard-surface-v2-run\steward_evaluation
```

## First held-out result

The corrected preregistered run completed 380,700 optimizer steps in 159.7
seconds. Its WSL cgroup receipt records 78.10 MB peak RAM, 47.29% CPU, and
17.86 MB/s peak block I/O against caps of 2,048 MB, 50%, and 50 MB/s. Swap was
disabled. Both held-out-target flags remained false, exit status was zero, and
the post-run audit found no lingering owned process or GPU compute application.

| Candidate | Parameters | Qualified | Pair F1 | Abstain recall | Array-status F1 | Exact arrays |
|---|---:|---:|---:|---:|---:|---:|
| Majority control | 0 | no | 0.1154 | 0.0000 | 0.3690 | 0.0000 |
| Forced-tradeoff control | 0 | no | 0.7069 | 0.0000 | 0.6563 | 0.5000 |
| Frozen interval policy | 0 | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Raw linear | 477 | no | 0.6877 | 0.4000 | 0.5471 | 0.0000 |
| Typed linear | 558 | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed latent, width 4 | 297 | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed looped, width 4, 2 loops | 313 | yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed looped, width 4, 4 loops | 313 | no | 0.9761 | 1.0000 | 0.8924 | 0.6667 |

Qualification requires perfect safety, orientation, and abstention behavior in
every seed. All qualified typed learned candidates also achieved exact selected
sets and exact rationales in every seed. The learned Pareto set contains only
the 297-parameter one-pass latent head: it ties the larger qualified models and
uses fewer parameters. Two loops add no measured value; four loops introduce a
seed-level regression.

This supports two narrow conclusions:

1. Explicitly typed uncertainty features are sufficient for this frozen rule,
   while the raw baseline does not reliably recover abstention.
2. Recurrence is not justified on this surface.

It does not show that the latent head discovered a new policy: the typed
features deliberately expose the relevant lattice deductions, and the frozen
policy control is exact by construction. It also does not authorize larger
model training, corpus-record promotion, or post-reveal tuning.

## Evidence bindings

- preregistration:
  `32140ce9c3da7b6364d5a5984ddfa388f532dabc754a1fdcec186b1bf2b1d727`;
- dataset manifest:
  `1a6794b4077ddc2c3deb72e1f0af09cee4931e6e2d5fcf4cb8ee274c8430f44a`;
- held-out predictions:
  `59e88121d4d92254a92d24557f7980da8493814eb060c594e1ff67d70539f57f`;
- trainer summary:
  `456665a527d291485e9187321dfa4af533c2444bc937912ce0b3119b2e9a46a2`;
- capped run receipt:
  `41690a5c2c5967c072bd57f8195885dcc1fa5c37915a9583c98f1db3a42057e9`;
- held-out evaluation:
  `d7babfed30fa60b9581943c033cc3ba54a8deeb4d535dd7178ff4d1ebfd399b5`;
- curation skill before:
  `111d1b136ea5866e499bd6a6990a59e7601985821874dc5e25eb6ad8317076c3`;
- curation skill after adding the confirmed uncertainty and array-reduction
  rules:
  `188ada227ef70b31ad8d37c9e72318714480e7b06a2dd55e4812f93c417cf868`.

## Next gate

Do not reuse this held-out split. Confirm the retained 297-parameter
architecture on a separately generated surface with variable array sizes,
asymmetric partial intervals, calibrated confidence scoring, and fresh
tournament-summary lineages. Only that confirmation can justify using its
recommendations as one input to curation tooling; authority remains with the
independent verifier.
