# LDT array confirmation v3

This experiment confirms the v2 one-pass typed latent architecture on a fresh,
harder surface. It uses content-free tournament summaries, variable candidate
array sizes, asymmetric and one-sided evidence bounds, novel held-out context
categories, and development-only confidence calibration.

The confirmation is still a synthetic meta-selection experiment. Its outputs
are shadow recommendations subordinate to the independent verifier; they cannot
promote corpus records.

## Frozen confirmation contract

[`configs/preregistrations/ldt_array_confirmation_v3.yaml`](../configs/preregistrations/ldt_array_confirmation_v3.yaml)
was frozen before generation. It fixes:

- 24 development arrays and 12 held-out arrays;
- four, five, and six candidates, equally represented in both splits;
- all directed comparisons within each array: 496 development and 248 held-out
  pairs;
- 120 development and 60 held-out content-free tournament summaries, with every
  pair bound to the hashes of its two source summaries;
- disjoint development and held-out lineages and context categories;
- both orientations for every unordered candidate pair; and
- environment-owned pair and array targets held outside the public candidate
  workspace.

A missing lower bound is conservatively completed to zero; a missing upper
bound is completed to one. A directional result is allowed only when dominance
survives those conservative bounds. Otherwise the policy abstains. This keeps a
strong one-sided lower bound useful without treating unsupported upper-only
evidence as complete.

## Models and calibration

The learned candidates are raw and typed linear heads, a width-four one-pass
typed latent head, and a width-four two-loop weight-tied head. All are trained
from scratch on the v3 development split; no v2 model or weight is loaded.

Each seed produces out-of-fold development predictions. A temperature is then
selected from the frozen grid `0.5, 0.75, 1.0, 1.5, 2.0, 3.0` by development
multiclass negative log likelihood. The selected temperature is bound into the
model, prediction, and trainer receipts. Steward labels never participate in
calibration.

Held-out evaluation reports multiclass Brier score, negative log likelihood,
and ten-bin expected calibration error. Qualification requires:

- perfect safety and safety-orientation behavior;
- perfect overall orientation consistency and abstention recall;
- expected calibration error no greater than 0.10 in every seed; and
- selected-set exact accuracy of at least 0.90 in every seed.

## Reproduce

Keep generated data, checkpoints, predictions, and labels outside the
repository:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_ldt_array_confirmation.py `
  --config configs/preregistrations/ldt_array_confirmation_v3.yaml `
  --out D:\Temp\datagoblin-ldt-array-confirmation-v3

.\scripts\run_ldt_array_confirmation_capped_wsl.ps1 `
  -DatasetManifest D:\Temp\datagoblin-ldt-array-confirmation-v3\public_dataset\dataset_manifest.json `
  -OutDir D:\Temp\datagoblin-ldt-array-confirmation-v3-run
```

After the run receipt passes, perform the single steward join:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_ldt_array_confirmation.py `
  --preregistration configs/preregistrations/ldt_array_confirmation_v3.yaml `
  --dataset-manifest D:\Temp\datagoblin-ldt-array-confirmation-v3\public_dataset\dataset_manifest.json `
  --predictions D:\Temp\datagoblin-ldt-array-confirmation-v3-run\capped_run\heldout_pair_predictions.jsonl `
  --trainer-summary D:\Temp\datagoblin-ldt-array-confirmation-v3-run\capped_run\trainer_summary.json `
  --pair-labels D:\Temp\datagoblin-ldt-array-confirmation-v3\private_steward\heldout_pair_labels.jsonl `
  --array-labels D:\Temp\datagoblin-ldt-array-confirmation-v3\private_steward\heldout_array_labels.jsonl `
  --out D:\Temp\datagoblin-ldt-array-confirmation-v3-run\steward_evaluation
```

## First held-out result

The capped run completed 510,000 optimizer steps in 180.3 seconds. Its cgroup
receipt recorded 79.38 MB peak and 73.61 MB average RAM, 47.44% CPU, and
17.64 MB/s peak block I/O against caps of 2,048 MB, 50%, and 50 MB/s. Swap was
disabled. Exit and WSL-stage cleanup statuses were zero, both held-out-target
flags remained false, and cleanup found no lingering owned process or GPU
compute application.

| Candidate | Parameters | Qualified | Pair F1 | Abstain | ECE | Status F1 | Exact arrays |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority control | 0 | no | 0.1242 | 0.0000 | 0.0000 | 0.3971 | 0.0000 |
| Forced-tradeoff control | 0 | no | 0.6613 | 0.0000 | 0.1774 | 0.6184 | 0.4167 |
| Frozen partial-interval policy | 0 | yes | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Raw linear | 630 | no | 0.8005 | 1.0000 | 0.0659 | 0.7465 | 0.0000 |
| Typed linear | 720 | yes | 1.0000 | 1.0000 | 0.0061 | 1.0000 | 1.0000 |
| Typed latent, width 4 | 369 | yes | 1.0000 | 1.0000 | 0.0135 | 1.0000 | 1.0000 |
| Typed looped, width 4, 2 loops | 385 | yes | 1.0000 | 1.0000 | 0.0143 | 1.0000 | 1.0000 |

The typed latent head was exact in every seed for four-, five-, and
six-candidate arrays. Its mean multiclass Brier score was 0.00324 and mean
negative log likelihood was 0.0148. The per-seed development calibration
temperatures were `1.0`, `0.75`, and `1.0`.

The learned Pareto set contains only `latent-typed-h4-v3`. The typed linear
head's calibration advantage remains inside the preregistered 0.01 equivalence
tolerance, while it uses 720 parameters. The two-loop head ties all predictive
and array metrics but uses 385 parameters and has no calibration advantage
outside tolerance. Recurrence again provides no evidence-backed benefit.

The majority control illustrates why calibration is only one qualifier: its
aggregate confidence happens to match its base-rate accuracy, producing zero
ECE, while its pair and whole-array behavior is unusable.

## Evidence bindings

- preregistration:
  `f2ef7459998587d609da8da0fe7749434c2e78a654977c0be0ba55a27fa5a985`;
- dataset manifest:
  `fb100fdce2de9af43002400cadf12e416e313dfce0174d06ef9fe7c8e101d485`;
- held-out predictions:
  `9680b5c44e1dd2e78a1f47228fb27c125c1a1aa95bbc759ed36e7cc272e102e2`;
- trainer summary:
  `75e1c59e425d8c8f9faa9dc8cf3a44c459dc91d65c017a48a8593f532b85e3d5`;
- capped run receipt:
  `cab482de86b5b0cf07ae427dad5f2d90ca627839f93e681a4585a8a16b7be629`;
- cleanup receipt:
  `2123dbe4dba1306f5f6aec9f92cd511add788d0c4c3290bdce2d9ddb4e94e40e`;
- steward pair labels:
  `908b359e745623f6d119cb8ad2cc108e78acecef5e29d184b0be8cb84f9db496`;
- steward array labels:
  `43cf5dfe0cdba95aa9fb38a01408812a0fb6061b87176fe61edae19e0f4de15b`;
- held-out evaluation:
  `73d0084f4d15daf3842321aba632fd4852e19798c55a52acbabbd0e2fe9b0ebd`;
- curation skill before:
  `188ada227ef70b31ad8d37c9e72318714480e7b06a2dd55e4812f93c417cf868`;
- curation skill after adding development-only calibration and variable-size
  confirmation guidance:
  `8ce580489fec9d2fd9d3a797c076f6c3ca11629746022a7731c0d925d7de48b5`.

## Interpretation and next boundary

This is a second fresh synthetic confirmation of the small typed one-pass
architecture. It still does not prove discovery of a new curation policy: the
typed features expose the relevant conservative lattice deductions, and the
environment-rule control is exact by construction.

The evidence authorizes an interface for shadow recommendations on fresh,
approved, content-free tournament summaries. A shadow receipt must retain the
model and calibration hashes, expose abstentions and unresolved candidates, and
compare its recommendation with the independent verifier without changing the
verifier's decision. No larger model, corpus promotion, or autonomous action is
authorized.

That interface and its first fresh synthetic shadow round are complete; see
[`ldt_shadow_v4.md`](ldt_shadow_v4.md).
