# LDT array sorter gym

This gym treats curation taste as a small, typed pairwise sorting problem. It compares
transparent controls, dense linear arrays, one-pass latent arrays, and weight-tied looped
arrays on the split-safe tournament curriculum produced by the curation-skill succession.
The sorter is always recommendation-only.

## Architecture

Each comparison has a public left and right adapter description. The encoder is fitted on
the training split only and emits:

- signed and absolute differences for ordinary and OOD exact mapping and mapping F1;
- safety and trainable-parameter differences;
- signed context and optimization indicators plus game and pressure indicators; and
- optional typed lattice deductions for safety direction, Pareto direction, metric
  equivalence, parameter-efficiency direction, and irreducible tradeoff.

The typed deductions form the explicit `a_t` state. Latent candidates transform that state
into a small private `h_t` array. Looped candidates repeatedly apply the same recurrent
matrix, so two and four loops at width four both have 223 trainable parameters; extra loops
add compute, not parameters. Both preference and rationale heads are trained jointly.

Every training comparison is augmented with its left/right swap while retaining the same
meta-lineage. This removes ordering as a training shortcut. Evaluation reports both
observed accuracy and swap consistency.

## Frozen experiment

[`configs/preregistrations/ldt_array_sorter_v1.yaml`](../configs/preregistrations/ldt_array_sorter_v1.yaml)
fixes:

- 180 environment-owned training pairs and 180 lineage-disjoint heldout inputs;
- three seeds and leave-one-training-lineage-out development evaluation;
- majority and exact-policy controls;
- raw and typed linear heads, width-four one-pass and two/four-loop heads, and a
  width-eight four-loop head;
- batches of at most 24 arrays, periodic checkpoints, and zero heldout-label access; and
- 2,048 MB RAM, 50% CPU, 50 MB/s block I/O, zero swap, private networking, and a
  900-second timeout under the WSL2 cgroup wrapper.

Generated plans, checkpoints, predictions, steward labels, and evaluations must stay
outside the repository.

## Run

Train with public inputs only:

```powershell
.\scripts\run_ldt_array_gym_capped_wsl.ps1 `
  -DatasetManifest D:\Temp\curation-succession\public_ldt_dataset\dataset_manifest.json `
  -OutDir D:\Temp\ldt-array-sorter-v1
```

Perform the one steward-owned heldout join:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_ldt_array_sorters.py `
  --dataset-manifest D:\Temp\curation-succession\public_ldt_dataset\dataset_manifest.json `
  --predictions D:\Temp\ldt-array-sorter-v1\capped_run\heldout_predictions.jsonl `
  --trainer-summary D:\Temp\ldt-array-sorter-v1\capped_run\trainer_summary.json `
  --heldout-labels D:\Temp\curation-succession\private_steward\heldout_labels.jsonl `
  --out D:\Temp\ldt-array-sorter-v1\steward_evaluation
```

If evaluator logic is corrected without changing models, predictions, labels, or
hyperparameters, preserve the original evaluation and pass it through
`--supersedes-evaluation` into a new output directory.

## First heldout result

The preregistered run completed 351,000 bounded optimizer steps in 137.8 seconds. The
cgroup receipt recorded a 70.45 MB RAM peak, 46.43% CPU, and 13.89 MB/s peak I/O; cleanup
found no lingering owned process or GPU compute application.

| Candidate | Parameters | Preference macro F1 | Safety | Tradeoff recall | Joint exact |
|---|---:|---:|---:|---:|---:|
| Majority control | 0 | 0.2571 | orientation-disqualified | 0.0000 | 0.2778 |
| Raw linear | 252 | 0.9475 | 1.0000 | 1.0000 | 0.8222 |
| Typed linear | 294 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed latent, width 4 | 207 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed looped, width 4, 2 loops | 223 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed looped, width 4, 4 loops | 223 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Typed looped, width 8, 4 loops | 471 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

All values are seed means; typed candidates were exact in all three seeds. The learned
Pareto set contains only the 207-parameter one-pass latent array because every larger
typed or looped model had equivalent heldout quality. Recurrence provided no measurable
benefit on this task.

The first evaluator report also exposed an orientation confound: the majority control
selected every observed safety comparison correctly because the safe candidate was always
on the same side, yet it had zero swap consistency. The immutable original report was
superseded by a logic-only correction requiring safety-row orientation consistency. No
model, prediction, label, or hyperparameter changed, so the heldout reveal count remains
one.

Evidence hashes:

- preregistration: `7ce6c0a327555bd3e5109525263d7577a38a04a39e2538339e71695c8b317df7`;
- dataset manifest: `b9bf20622c2cb103f496bbd1e4ea2f03abd88c2ca3dc3596684e3fbee3ac72f6`;
- heldout predictions: `09a6d516bbb9a8c07f6d5d12fe3e658b23d78888ac2f739e42d506a0ad859538`;
- superseded evaluation: `99acca2c015144547f01e353ddecdb4db577baffbcc24649112cf025dd2ac2d1`;
- corrected evaluation: `62b8ca076761a44bd5104d5e9c3cb292b648f0969c076aa05f2876d474333d98`;
- curation skill before: `4b24b8c641979c7ca00bd89fad70f4c22bb185a1ed36ed7fdcb5cd2936ad97aa`;
- curation skill after the orientation-audit rule:
  `111d1b136ea5866e499bd6a6990a59e7601985821874dc5e25eb6ad8317076c3`.

## Interpretation and next gate

This is evidence that a tiny typed array head can reproduce the explicit curation rule,
not evidence that it discovered a new policy. The typed encoder nearly states the rule,
and the frozen policy control is exact by construction. A larger checkpoint or more loops
would add cost without addressing the main uncertainty.

The next useful tournament should make the sorting surface harder while keeping authority
unchanged: balance left/right orientation in the emitted dataset, add near-tolerance
boundaries, missing or interval-valued metrics, unknown context categories, and
multi-comparison arrays where a local decision can conflict with a globally nondominated
set. That fresh succession needs new development and heldout lineages; this heldout split
must not be reused for tuning.
