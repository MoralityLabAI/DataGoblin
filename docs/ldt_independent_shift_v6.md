# Independent LDT distribution shift v6

This gate tests whether the earlier LDT result is architecture evidence or
whether it depends on typed features derived by executing the verifier policy.
It uses an independent, target-free fixture builder, seals verifier labels
before candidate inference, freezes every candidate output before the single
join, and compares the same checkpoint with and without its ten policy-lattice
features.

Model weights, public fixture rows, private verifier labels, candidate outputs,
and generated comparisons remain outside the repository.

## Why this audit was necessary

The v3 typed encoder includes ten `lattice:*` features. They encode safety
direction, robust dominance, metric equivalence, efficiency direction,
incomplete evidence, partial bounds, interval width, insufficiency, and
irreducible tradeoff. Those features are computed by invoking the exact taste
policy used to define the synthetic targets.

The earlier shadow runs therefore tested a deterministic rule-plus-model stack.
They did not isolate what the latent architecture learned from non-policy
inputs.

## Frozen contract and custody

[`configs/preregistrations/ldt_independent_shift_v6.yaml`](../configs/preregistrations/ldt_independent_shift_v6.yaml)
binds:

- the v5 succession comparison;
- the original v3 trainer summary and three frozen seed-17 checkpoints;
- an independent builder that never imports or emits verifier targets;
- a separate environment-owned verifier implementation;
- four new pressure families, seed 3203, and all 24 prior lineage hashes;
- 28 missing interval bounds and at least 20 near-tolerance comparisons;
- five setup definitions, all ineligible for model selection;
- one sealed verifier reveal; and
- no training, promotion, deployment, or decision authority.

The contract hash is
`df86d47a53c5f51a75905f21e9276146630e786bacfc0ae5a030ccd379a6da2e`.

The five setups are:

1. the frozen latent checkpoint with the full policy lattice, retained only as
   a circularity diagnostic;
2. the identical latent checkpoint with only the ten `lattice:*` inputs
   replaced by zero;
3. the frozen typed linear checkpoint with the same lattice mask;
4. the frozen raw-feature linear checkpoint; and
5. a zero-parameter safety-first scalar rule.

The same-checkpoint latent comparison preserves model weights, calibration,
input width, and all non-lattice preprocessing. Only the policy-derived
features change.

## Run

Generate public, target-free inputs:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_ldt_independent_shift.py `
  --contract configs/preregistrations/ldt_independent_shift_v6.yaml `
  --out D:\Temp\datagoblin-ldt-independent-shift-v6-fixture
```

Seal verifier labels before candidate inference:

```powershell
python scripts/seal_ldt_shift_verifier.py `
  --contract configs/preregistrations/ldt_independent_shift_v6.yaml `
  --shift-manifest `
    D:\Temp\datagoblin-ldt-independent-shift-v6-fixture\public_shift\shift_manifest.json `
  --out D:\Temp\datagoblin-ldt-independent-shift-v6-verifier
```

Run all candidates without passing the verifier directory:

```powershell
python scripts/run_ldt_shift_candidates.py `
  --contract configs/preregistrations/ldt_independent_shift_v6.yaml `
  --shift-manifest `
    D:\Temp\datagoblin-ldt-independent-shift-v6-fixture\public_shift\shift_manifest.json `
  --model-root D:\external-confirmation-run\models `
  --trainer-summary D:\external-confirmation-run\trainer_summary.json `
  --out D:\Temp\datagoblin-ldt-independent-shift-v6-run
```

Perform the single verifier join:

```powershell
python scripts/evaluate_ldt_independent_shift.py `
  --contract configs/preregistrations/ldt_independent_shift_v6.yaml `
  --shift-manifest `
    D:\Temp\datagoblin-ldt-independent-shift-v6-fixture\public_shift\shift_manifest.json `
  --candidate-run-dir D:\Temp\datagoblin-ldt-independent-shift-v6-run `
  --verifier-dir D:\Temp\datagoblin-ldt-independent-shift-v6-verifier `
  --out D:\Temp\datagoblin-ldt-independent-shift-v6-run\verifier_comparison
```

## Shift surface

The builder produced 248 orientation-balanced pairs, 12 arrays, and 60
content-free summaries. It generated four new lineage groups with no overlap
against the 24 forbidden v3-v5 groups.

The surface contains:

- 96 directed metric comparisons between 0.008 and 0.013 apart;
- 28 missing interval bounds;
- four complete, non-point interval metrics;
- 16 `ABSTAIN`, 56 `TRADEOFF`, 88 `LEFT`, and 88 `RIGHT` pair targets; and
- all four array statuses.

The candidate workspace received no target or verifier path. All five setup
receipts and prediction files were hash-bound before the join.

## Results

| Setup | Pair accuracy | Macro F1 | Abstain recall | Swap | ECE | Exact arrays |
|---|---:|---:|---:|---:|---:|---:|
| Full-lattice latent diagnostic | 1.000 | 1.000 | 1.000 | 1.000 | 0.067 | 1.000 |
| Lattice-masked latent | 0.613 | 0.464 | 0.500 | 0.903 | 0.160 | 0.000 |
| Lattice-masked typed linear | 0.605 | 0.426 | 0.250 | 0.968 | 0.116 | 0.000 |
| Frozen raw linear | 0.665 | 0.642 | 0.938 | 0.968 | 0.136 | 0.250 |
| Non-learned safety scalar | 0.815 | 0.772 | 1.000 | 1.000 | 0.184 | 0.250 |

Every setup preserved safety-gate accuracy of `1.0`. Only the circular
diagnostic and the non-learned baseline passed all frozen monitoring
thresholds. Passing those thresholds is not model selection; every setup
remains explicitly ineligible.

Masking the policy-derived features from the same latent checkpoint caused:

- preference accuracy to fall by `0.3871`;
- support-aware preference macro F1 to fall by `0.5360`;
- exact-array accuracy to fall by `1.0`; and
- ECE to increase by `0.0923`.

The masked latent produced 96 preference disagreements, failed swap
consistency, and marked no array exactly. The non-learned rule made 46 pair
errors and outperformed every non-circular learned setup on pair accuracy and
macro F1. The raw linear model had the best non-circular learned result, but
failed the exact swap gate and was exact on only three of 12 arrays.

The hash-only review queue contains 391 setup-disagreement keys:

- 126 for the masked latent;
- 123 for the masked typed linear;
- 96 for the raw linear; and
- 46 for the non-learned baseline.

It contains slice metadata and mismatch axes, never verifier labels.

## Evidence bindings

- public shift manifest:
  `572539d27e2f00034681d41709303ca3a9f024285607932969193a948e2b6898`;
- verifier seal receipt:
  `a5aea4b221ce518d804b9c849ded96153d43abdfc671d17a6911da88cb775c7e`;
- sealed pair labels:
  `9ecf5d95184241ee1b72ddfb6c5b8c6068c43a13e9a75356afe1c72f8a158dc8`;
- sealed array labels:
  `d94b4dc61cf90115436a96e182532fa8809dbdd958ba68327d636c362850f0de`;
- aggregate candidate-run receipt:
  `bf9f28a11f736a821713f53e32ac45efaa83ba752969d9a3ca3bcd5c53cc69c0`;
- immutable comparison:
  `fe9af976a854e667b6544c672a9538b0ee7afa55d38b25b1eb6ab58963964607`;
- comparison receipt:
  `9b8a7417c300996c97c779901731c2b70c5fc5d57b609462d917b520484fa71d`;
  and
- hash-only review queue:
  `f6b578df0fbe23de68b24399aee7d5daedfeb55954599d8d8d1d68e7938afa59`.

## Evidence interpretation

The policy-lattice reliance criterion passed decisively. The v3-v5 results
remain valid evidence that the complete typed policy-plus-head stack is
deterministic, safe on those synthetic surfaces, and well integrated with
verifier custody. They are not evidence that the 369-parameter latent
architecture independently learned the sorting policy.

In particular, the earlier low ECE measured the circular full-stack interface.
It should not be used to justify LoRA training, HRM-Text training, a larger
checkpoint, or deployment.

This finding changed the curation skill. Its tournament guidance now requires:

- a feature-derivation audit before architecture claims;
- diagnostic-only treatment for target- or verifier-policy-derived features;
- a same-checkpoint policy-feature mask;
- raw-linear and non-learned controls; and
- reclassification of a masked collapse as policy-feature reliance.

The skill fingerprint changed from
`01bc231813cd3300974a9b6bf87091e41d05bf624c38508915546c15b5b6663f`
to
`a08298ed946b798612f2fe09fabdecd647bf013757f82dc0118acedd4e45ad56`.

## Next gate

Do not train from the sealed v6 labels or its disagreement queue. The next
valid experiment is a development-only feature redesign that excludes all
verifier-policy outputs and target proxies. It should train bounded raw or
independently typed models against development lineages, compare them with the
non-learned rule, freeze calibration on development predictions, and then use
new sealed lineages for one confirmation reveal.
