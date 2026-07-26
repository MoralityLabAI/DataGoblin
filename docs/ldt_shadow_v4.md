# LDT shadow integration v4

This gate runs the confirmed one-pass typed latent sorter beside the independent
curation verifier without allowing the model to affect any decision. The runner
accepts only content-free tournament summaries, loads an operator-supplied
external model by hash, and emits recommendation-only pair and array receipts.

Model weights, generated fixtures, verifier labels, and run outputs remain
outside the repository.

## Frozen contract

[`configs/preregistrations/ldt_shadow_v4.yaml`](../configs/preregistrations/ldt_shadow_v4.yaml)
binds:

- the v3 confirmation preregistration, manifest, and evaluation hashes;
- `latent-typed-h4-v3`, seed 17, with 369 parameters;
- model hash
  `4ea5efd01de222279dc02e942fa95f6a944caa52ec20889fe1580e6867a3e94b`;
- trainer-summary hash
  `75e1c59e425d8c8f9faa9dc8cf3a44c459dc91d65c017a48a8593f532b85e3d5`;
- temperature `1.0`, fitted from out-of-fold development predictions;
- every prior v3 development and held-out lineage hash; and
- `decision_effect: NONE`, environment-owned verifier authority, and no
  training or corpus-promotion authority.

Seed 17 is selected because it is the lowest preregistered seed. It is not
selected from held-out or shadow performance.

## Interface

The fixture contains 248 directed pairs, 12 arrays, and 60 content-free
tournament summaries. Four, five, and six candidates are equally represented.
Public inputs contain no verifier targets; private pair and array labels remain
in a separate verifier directory.

The runner:

1. checks the contract, model, trainer, encoder, and calibration hashes;
2. rejects recurrent, malformed, non-finite, or repository-resident model
   states;
3. validates every tournament-report binding;
4. emits calibrated pair recommendations;
5. reduces them into `SELECTED`, `DOMINATED`, `UNRESOLVED`, and `UNSAFE`
   array statuses; and
6. writes an immutable receipt stating that verifier targets were unseen and
   the decision effect was none.

The independent join revalidates those artifacts before opening verifier
custody. Its review queue contains hashes and aggregate slice metadata only,
never verifier labels.

## Run

Generate a fresh content-free shadow fixture:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_ldt_shadow_fixture.py `
  --contract configs/preregistrations/ldt_shadow_v4.yaml `
  --out D:\Temp\datagoblin-ldt-shadow-v4-fixture
```

Run the frozen external model without verifier labels:

```powershell
python scripts/run_ldt_shadow.py `
  --contract configs/preregistrations/ldt_shadow_v4.yaml `
  --shadow-manifest D:\Temp\datagoblin-ldt-shadow-v4-fixture\public_shadow\shadow_manifest.json `
  --model D:\external-confirmation-run\models\latent-typed-h4-v3-seed-17.json `
  --trainer-summary D:\external-confirmation-run\trainer_summary.json `
  --out D:\Temp\datagoblin-ldt-shadow-v4-run
```

Perform the independent verifier join:

```powershell
python scripts/evaluate_ldt_shadow.py `
  --contract configs/preregistrations/ldt_shadow_v4.yaml `
  --shadow-manifest D:\Temp\datagoblin-ldt-shadow-v4-fixture\public_shadow\shadow_manifest.json `
  --pair-recommendations D:\Temp\datagoblin-ldt-shadow-v4-run\shadow_pair_recommendations.jsonl `
  --array-recommendations D:\Temp\datagoblin-ldt-shadow-v4-run\shadow_array_recommendations.jsonl `
  --shadow-run-receipt D:\Temp\datagoblin-ldt-shadow-v4-run\shadow_run_receipt.json `
  --verifier-pair-labels D:\Temp\datagoblin-ldt-shadow-v4-fixture\private_verifier\verifier_pair_labels.jsonl `
  --verifier-array-labels D:\Temp\datagoblin-ldt-shadow-v4-fixture\private_verifier\verifier_array_labels.jsonl `
  --out D:\Temp\datagoblin-ldt-shadow-v4-run\verifier-comparison
```

## Round 1 result

The shadow runner produced 248 pair and 12 array recommendations. It emitted 44
pairwise abstentions and marked 20 candidates unresolved. It read no verifier
targets or raw game records, used no network, persisted no weights to the
repository, and recorded no decision effect.

The independent comparison passed every frozen monitoring gate:

| Metric | Result |
|---|---:|
| Preference macro F1 | 1.0000 |
| Preference accuracy | 1.0000 |
| Abstain recall | 1.0000 |
| Safety-gate accuracy | 1.0000 |
| Swap consistency | 1.0000 |
| Expected calibration error | 0.01893 |
| Multiclass Brier | 0.00257 |
| Negative log likelihood | 0.01987 |
| Array-status macro F1 | 1.0000 |
| Exact arrays | 1.0000 |
| Selected-set exact accuracy | 1.0000 |
| Hash-only review queue | 0 |

Each game contributed 62 pair recommendations and three arrays. Pair F1 and
array exact accuracy were 1.0 for Centauri, Court, Diplomacy, and They Sing.
ECE ranged from 0.0142 to 0.0211. All four-, five-, and six-candidate array
slices were exact.

The original comparison exposed a metric-reporting defect: the exact
six-candidate slice showed macro-F1 0.75 because an absent `UNSAFE` class was
included as zero support. The immutable original report was superseded by
revision 2, which uses support-aware macro-F1 for slices. Model weights,
recommendations, labels, thresholds, and the single verifier join did not
change.

## Evidence bindings

- shadow contract:
  `7416a92f49c6040630c1843f0459d38ea0c2abf70448b527124dc0116818a08b`;
- public shadow manifest:
  `811b5e6c7f72a0549464ffa936b796fbbeb6d3887a130ebfb8043cee51e099ea`;
- verifier pair labels:
  `103de4de4c04da91027206b9194707bc823375de1691f3ac27dfe1b6a1a2914f`;
- verifier array labels:
  `9e2ccc1f67683ef77d473a801dbcaef30469896b070aacb6bad5834cea242815`;
- shadow run receipt:
  `93bad0de1af049180b15747b1839a321cb8d32227a8e6a1738a16414ee86efc2`;
- pair recommendations:
  `ae13f340eb2aa567debb732d35beb6404c3b60555bf48bd8b2b97f056e07a220`;
- array recommendations:
  `866b98195031fd10ed42df15bedc890f31f19d8e55d9d04ebfa9d95c343db83a`;
- immutable original comparison:
  `773604ee270d74fbc4459a30cea1fa8f3787560f5ae42bff3154d2d1cca02e9e`;
- corrected comparison revision 2:
  `3f19925fba58a6b5b8ddacd91aec7ceafd3b98b8b58a5d1ec9bb4e2581fde5bc`;
- empty hash-only review queue:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- curation skill before:
  `8ce580489fec9d2fd9d3a797c076f6c3ca11629746022a7731c0d925d7de48b5`;
- curation skill after support-aware slice guidance:
  `01bc231813cd3300974a9b6bf87091e41d05bf624c38508915546c15b5b6663f`.

## Boundary and next gate

Round 1 validates the shadow interface on a fresh synthetic surface. It is not
a deployment claim and does not authorize training from disagreements—there
were no disagreements to review.

The next evidence-producing step is another append-only shadow round on a
separately approved content-free tournament-summary export with new lineages.
The independent verifier remains the only decision owner. Any disagreement
sample requires separate authorization before content-free review and still
cannot expose raw records or verifier labels to the model runner.
