# LDT shadow succession v5

This gate tests whether the confirmed one-pass typed latent sorter remains
stable across three append-only shadow rounds. Every round uses a new synthetic
seed and new content-free tournament-summary lineages. The frozen seed-17 model
and its development-only temperature remain unchanged.

Model weights, generated fixtures, private verifier labels, recommendations,
and comparison outputs remain outside the repository.

## Frozen contract

[`configs/preregistrations/ldt_shadow_succession_v5.yaml`](../configs/preregistrations/ldt_shadow_succession_v5.yaml)
binds:

- the corrected v4 comparison and its single verifier join;
- `latent-typed-h4-v3`, seed 17, with 369 parameters;
- model hash
  `4ea5efd01de222279dc02e942fa95f6a944caa52ec20889fe1580e6867a3e94b`;
- trainer-summary hash
  `75e1c59e425d8c8f9faa9dc8cf3a44c459dc91d65c017a48a8593f532b85e3d5`;
- the unchanged temperature `1.0`, fitted only on out-of-fold development
  predictions;
- three new fixture seeds: 1901, 2017, and 2131;
- all 12 prior v3 and v4 lineage hashes;
- cross-round lineage disjointness and one independent join per round; and
- no decision, promotion, training, model-selection, or deployment authority.

The contract hash is
`4bb3ff84151f52a473e0035ddf3e1e1a7449aae7ce5451357dff3977a908d095`.

## Append-only custody flow

The generator derives each round contract before generating that round. Round
2 excludes the 12 preregistered prior lineages. Rounds 3 and 4 also exclude the
four lineage groups observed in each earlier succession round. The public
succession manifest contains public artifact paths and hashes, but no private
verifier paths or targets.

The runner opens only the derived contracts, public manifests, public
content-free reports, and the operator-supplied external model. It emits
recommendation-only artifacts with `decision_effect: NONE`.

The independent evaluator then opens verifier custody once for each round. It
writes one immutable v4 comparison per round and a v5 aggregate derived only
from those comparisons. The combined review queue is hash-only.

## Run

Generate all three fixtures outside the repository:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_ldt_shadow_succession.py `
  --contract configs/preregistrations/ldt_shadow_succession_v5.yaml `
  --out D:\Temp\datagoblin-ldt-shadow-succession-v5-fixture
```

Run the frozen external model without verifier labels:

```powershell
python scripts/run_ldt_shadow_succession.py `
  --contract configs/preregistrations/ldt_shadow_succession_v5.yaml `
  --succession-manifest `
    D:\Temp\datagoblin-ldt-shadow-succession-v5-fixture\public_succession\succession_manifest.json `
  --model D:\external-confirmation-run\models\latent-typed-h4-v3-seed-17.json `
  --trainer-summary D:\external-confirmation-run\trainer_summary.json `
  --out D:\Temp\datagoblin-ldt-shadow-succession-v5-run
```

Perform the independent joins and comparison-only aggregation:

```powershell
python scripts/evaluate_ldt_shadow_succession.py `
  --contract configs/preregistrations/ldt_shadow_succession_v5.yaml `
  --succession-manifest `
    D:\Temp\datagoblin-ldt-shadow-succession-v5-fixture\public_succession\succession_manifest.json `
  --shadow-run-dir D:\Temp\datagoblin-ldt-shadow-succession-v5-run `
  --out D:\Temp\datagoblin-ldt-shadow-succession-v5-run\verifier_comparison
```

## Results

Generation produced 744 directed pairs, 36 arrays, and 180 content-free
tournament summaries. The 12 new lineage groups were mutually disjoint and had
zero overlap with the 12 excluded v3/v4 groups. Later derived contracts
increased their lineage exclusion counts from 12 to 16 to 20.

The public-only runner emitted 132 abstentions and 60 unresolved candidates
across the three rounds. It recorded no verifier-target access, raw-record
access, network use, repository weight persistence, or decision effect.

Every independent monitoring comparison passed:

| Round | Pair macro F1 | ECE | Brier | NLL | Exact arrays | Reviews |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1.0000 | 0.01900 | 0.00258 | 0.01993 | 1.0000 | 0 |
| 3 | 1.0000 | 0.01897 | 0.00258 | 0.01990 | 1.0000 | 0 |
| 4 | 1.0000 | 0.01896 | 0.00258 | 0.01989 | 1.0000 | 0 |

Worst-case succession metrics were:

- pair preference macro F1: `1.0`;
- abstain recall: `1.0`;
- expected calibration error: `0.01899693907577777`;
- multiclass Brier: `0.0025811225966087888`;
- multiclass negative log likelihood: `0.01993340470858161`;
- exact-array and selected-set accuracy: `1.0`; and
- disagreement-review keys: `0`.

Every game had pair macro F1 and exact-array accuracy of `1.0` in every round.
Per-game ECE ranges were:

| Game | Minimum ECE | Maximum ECE |
|---|---:|---:|
| Centauri | 0.01827 | 0.02105 |
| Court | 0.02070 | 0.02343 |
| Diplomacy | 0.02192 | 0.02194 |
| They Sing | 0.01215 | 0.01236 |

Each verifier join count is exactly one, all verifier decisions remain
immutable, and the combined hash-only review queue is empty.

## Evidence bindings

- public succession manifest:
  `07dc9286f357961445bb08c2df0371372d47dba338b2468bf51a13840b51b86b`;
- aggregate public-only run receipt:
  `01ae65901e4cca11714667a0f80024553b5c557f32b3e2065b86866e855a8fd0`;
- round 2, 3, and 4 comparisons:
  `9b47e8f191b44b0620a14d78449433e546ca5fd6461195ba90621e3026d046eb`,
  `62cc8d24d38c74daa800cc2310a38cd03c3630cf7e9a4e07b454fd011f6b1617`,
  and
  `8bc0fc227b3e3b72a2e35144e6c65197b4404ddbf42910cffa2b0a47990d7e7e`;
- aggregate succession comparison:
  `1d2ba6912ee98cc3e34c5db4730ecb1afdd672d40746ffe6c4b2d3af64437fb4`;
- aggregate comparison receipt:
  `04b6fd47ca8683739709cf0a351bead3daa9d277cd29401391251bf321f764df`;
- empty combined review queue:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
  and
- unchanged curation-skill fingerprint:
  `01bc231813cd3300974a9b6bf87091e41d05bf624c38508915546c15b5b6663f`.

The skill did not change because this succession produced no new failure mode:
its existing rules already required fresh lineages, support-aware slices,
verifier separation, append-only corrections, and no training on sealed
labels.

## Interpretation and next gate

The frozen model's low ECE is stable on three additional synthetic
content-free surfaces. This is stronger evidence for deterministic interface
stability, not evidence of transfer to natural agent traces. The fixture
builder and verifier share the same typed synthetic task semantics, so exact
performance cannot justify deployment, retraining, or choosing this adapter
over alternatives.

That gate is completed by the
[`independent LDT distribution shift v6`](ldt_independent_shift_v6.md).
Its same-checkpoint ablation shows that the exact latent result depends on ten
features computed from the verifier policy. The v5 result remains interface
evidence, but it is not independent latent-architecture evidence.
