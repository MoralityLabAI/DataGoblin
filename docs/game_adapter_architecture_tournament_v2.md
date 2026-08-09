# Four-game adapter architecture tournament v2

V2 is the first downstream gate authorized by the independently confirmed edge-exact LDT v15. It is frozen before
official task generation. No result is recorded yet.

The scoped deterministic-development preflight returned `warn` with no blocking finding. Its only remaining warning is
the generic adapter's request for subjective completeness, correctness, epistemic-honesty, and style dimensions; those
dimensions are explicitly outside this exact environment-target claim. The frozen hashes are:

- tournament preregistration: `16e4dfd60b6cf6f9d898d8890df302725f75dfec5e1510925a373d2b82e977f2`;
- evaluation-review manifest: `2858a2b362179b0f356f8da8b75506c57e32857dfad7f4c68d16f05a6aaea7a8`;
- deterministic-development policy: `ec9e4776a88c7c182fcbf4411151dab7dcb7e89565538161d481097bfbaab682`;
- external preflight report: `a617de6c79af85d2d01e47adea9302a172819a60b77700dd9a83e480063f8358`.

The tournament compares four executable, offline candidates on the same shape-only Centauri, They Sing, Court, and
Diplomacy fixtures:

- the unchanged exact-key policy, with no learned parameters;
- a frozen canonical-field prototype plus a rank-4 low-rank update;
- a frozen recurrent reservoir with a learned readout and one update; and
- the matched frozen reservoir with four tied state updates, used as an HRM-Text-style structural surrogate.

The two trainable names are deliberately narrow. The low-rank arm is not a language-model LoRA run, and the recurrent
arm is not an HRM-Text checkpoint. They test whether those parameterization ideas add useful signal on a small task
before any operator considers supplying real weights.

## Evidence firewall

Three development surfaces and three heldout surfaces independently cover balanced schema drift, compound boundary
precedence, and evidence authority. Each surface contributes one candidate array per game, so development and heldout
each contain 12 exact whole-array units. The lanes use different generator seeds, and generation fails on a shared
lineage or exact record hash.

Trainable candidates receive only:

- the six public shape-only fixture manifests;
- environment-owned labels for the three development `train` splits; and
- environment-owned labels for the three development `calibration` splits.

Development evaluation labels and every heldout label remain in the private steward workspace. All candidate
predictions must exist before the no-label packet audit can authorize the single heldout join. The exact-key control
does not read even the development training release.

The fail-closed decision table remains a fixed component. Learned parameters map only safe public field aliases. This
means privacy routing is not credited to the learned architecture; it is a hard invariant that every arm must preserve.

## Matched training and resource contract

Each trainable architecture runs at seeds 31, 73, and 127. Seed outcomes become min/max metric intervals; there is no
best-seed substitution. The one-pass and four-loop recurrent arms use the same hidden width, fixed input and recurrent
matrices, learned-readout parameter count, data, seeds, and optimizer. Their declared intervention is the number of
tied state updates.

Every trainable cell must run in the existing WSL cgroup wrapper under 2,048 MB RAM, 50% CPU, 50 MB/s I/O, zero swap,
and 900 seconds. Checkpoints are due every 100 steps or 300 seconds, whichever occurs first. Each run preserves its cap,
checkpoint, abort, and PID-scoped cleanup receipts. No network or external checkpoint is permitted.

## Decision order

Safety gates are applied before utility. A candidate must have zero privacy leakage, zero critical false release, and
perfect sealed routing for every game, surface, and seed. Quality is then represented by per-game intervals for ordinary
exact record accuracy, ordinary mapping F1, OOD exact record accuracy, and OOD mapping F1. The environment applies
conservative interval Pareto dominance and uses parameter cost only when all quality axes are equivalent within 0.01.
It emits `UNSAFE`, `DOMINATED`, `UNRESOLVED`, or `SELECTED`; no scalar champion is computed.

The retained v15 model is bound by SHA-256 and consumes only the 12 content-free development receipt arrays. Its
predictions are sealed before heldout metrics are computed. The environment then derives the 12 authoritative heldout
status arrays. Confirmation requires at least 0.65 exact status-vector accuracy, 0.75 exact selected-set accuracy, the
frozen worst-surface floors, all candidate custody/resource gates, and at least one environment-qualified trainable
recipe with meaningful Pareto membership. The LDT remains `MODEL_RECOMMENDATION_ONLY`; its output cannot override an
environment status.

## Commands

Generated artifacts must stay outside the repository. After the harness is committed and the evaluation preflight
passes:

```powershell
$env:PYTHONPATH = "src"
python scripts/game_adapter_architecture_tournament_v2.py generate `
  --config configs/preregistrations/game_adapter_architecture_tournament_v2.yaml `
  --out D:\Temp\datagoblin-adapter-architecture-v2-fixture

scripts/run_game_adapter_architecture_tournament_v2_matrix.ps1 `
  -DatasetManifest D:\Temp\datagoblin-adapter-architecture-v2-fixture\public_dataset\tournament_manifest.json `
  -ReleaseManifest D:\Temp\datagoblin-adapter-architecture-v2-fixture\candidate_release\release_manifest.json `
  -OutRoot D:\Temp\datagoblin-adapter-architecture-v2-matrix
```

Development evaluation, v15 shadow inference, the no-label packet audit, the one heldout join, and the separate final
audit are separate subcommands or scripts. This separation is part of the evidence contract, not optional ceremony.

## Claim boundary

A passing result would admit only the named tiny recipe or recipes to a separately frozen next experiment. It would not
authorize real checkpoint training, corpus release, gameplay claims, deployment, or an architecture-superiority claim.
