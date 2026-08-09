# Four-game adapter architecture tournament v2

## Result

V2 completed with a valid negative promotion decision. The rank-4 low-rank surrogate was the only trainable arm to
clear both the development and heldout environment gates, but the retained v15 LDT recovered 0 of 12 heldout status
vectors and 0 of 12 heldout selected sets. The preregistered whole-array gate therefore failed and no candidate
advances. The single heldout reveal is spent; post-reveal tuning is false.

This separates two claims that must not be collapsed. The environment evidence supports the low-rank surrogate on this
synthetic shape task. It does not support using v15 as a shadow sorter for the new safe-only, seed-interval adapter
domain, and the frozen tournament required both layers before advancement.

The scoped deterministic-development preflight returned `warn` with no blocking finding. Its only remaining warning is
the generic adapter's request for subjective completeness, correctness, epistemic-honesty, and style dimensions; those
dimensions are explicitly outside this exact environment-target claim. The frozen hashes are:

- tournament preregistration: `16e4dfd60b6cf6f9d898d8890df302725f75dfec5e1510925a373d2b82e977f2`;
- evaluation-review manifest: `2858a2b362179b0f356f8da8b75506c57e32857dfad7f4c68d16f05a6aaea7a8`;
- deterministic-development policy: `ec9e4776a88c7c182fcbf4411151dab7dcb7e89565538161d481097bfbaab682`;
- external preflight report: `a617de6c79af85d2d01e47adea9302a172819a60b77700dd9a83e480063f8358`.

## Development evidence

All 10 candidate cells passed public preflight. Every trainable cell completed under the hard-cap wrapper with 10
checkpoints, passing cleanup, no network, no external checkpoint, and neither development-evaluation nor heldout labels
staged. All candidates had zero privacy leakage, zero critical false release, and perfect sealed routing.

| Candidate | Parameters | Ordinary exact | OOD exact | Development qualified |
|---|---:|---:|---:|:---:|
| exact-key control | 0 | 0.1646 | 0.3659 | no |
| rank-4 low-rank update | 1,116 | 0.9921 | 0.7777 | yes |
| one-pass recurrent readout | 759 | 0.9154 | 0.5066 | no |
| four-loop tied recurrent readout | 759 | 0.9143 | 0.5176 | no |

The recurrent arms cleared their aggregate ordinary and OOD floors but failed the frozen conservative every-game OOD
gain over the exact-key control. No best seed was substituted: each public feature is the min/max interval over all
three matched seeds.

The v15 shadow packet was then frozen before confirmation. It predicted the exact-key control as `DOMINATED` and all
three trainable candidates as `SELECTED` in all 12 arrays. The public packet audit passed every hash, candidate,
resource, contamination, authority, and no-label-access check before authorizing the single join.

## Independent heldout result

The low-rank surrogate repeated its environment qualification. The one-pass and four-loop recurrent arms again failed
the every-game conservative OOD-gain gate. All arms remained safety-clean.

| Candidate | Ordinary exact | OOD exact | Selected | Dominated | Unresolved | Environment qualified |
|---|---:|---:|---:|---:|---:|:---:|
| exact-key control | 0.1646 | 0.3659 | 0 | 12 | 0 | no |
| rank-4 low-rank update | 0.9980 | 0.7894 | 6 | 0 | 6 | yes |
| one-pass recurrent readout | 0.9104 | 0.5537 | 0 | 8 | 4 | no |
| four-loop tied recurrent readout | 0.8983 | 0.5677 | 0 | 7 | 5 | no |

The recurrent intervention is a tradeoff, not a win: four loops raised aggregate OOD exactness from 0.5537 to 0.5677
but reduced ordinary exactness from 0.9104 to 0.8983, with a Diplomacy OOD regression offsetting gains elsewhere.

| Frozen sorter gate | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| exact status-vector accuracy | 0.65 | 0.00 | no |
| exact selected-set accuracy | 0.75 | 0.00 | no |
| worst-surface exact status vector | 0.25 | 0.00 | no |
| worst-surface exact selected set | 0.25 | 0.00 | no |
| mean safety-status accuracy | diagnostic | 1.00 | n/a |

V15 correctly recognized that every entrant was safety-qualified and that the exact-key control was dominated. It
failed the material distinction among `SELECTED`, `DOMINATED`, and `UNRESOLVED` for the three trainable candidates.
Its previous synthetic confirmation therefore did not transport to this receipt distribution.

The independent final auditor reproduced every whole-array metric, every candidate advancement decision, all custody
and resource bindings, the single reveal, and the no-retune declaration. Its audit passed with conclusion
`not_confirmed_no_promotion`.

## Resource evidence

| Trainable arm, three seeds | Trainer seconds | Steps | Checkpoints | Peak RAM | Peak sampled CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|---:|
| rank-4 low-rank update | 9.79 | 2,880 | 30 | 70.36 MB | 50.81% | 15.292 MB/s |
| one-pass recurrent readout | 7.52 | 2,880 | 30 | 29.46 MB | 51.09% | 0 MB/s |
| four-loop tied recurrent readout | 7.44 | 2,880 | 30 | 28.71 MB | 50.61% | 0 MB/s |

The sampled CPU estimates can straddle 50% over short windows; every unit independently recorded the enforced systemd
`CPUQuota=50%`, 2,048 MB memory ceiling, 50 MB/s I/O ceilings, zero swap, and passing PID-scoped cleanup.

## Bound receipts

- frozen harness commit: `98784f36d517fb06bbc665cbd117981c76d400aa`;
- pre-training WSL path correction: `208f968`;
- negative-result auditor correction: `85f545f`;
- generation report: `c78394a4d647e9caf14fdfe1c5c0fffc75d3d1d2fec7d912e9a9d29b5adf356f`;
- public tournament manifest: `88c181279455b96c1d8a2eadc69ff1cfa581cd8fb1c23e2a5ddaf466bfb70c1d`;
- development-only training release: `831cc9d4e90651b594901b0ea3ee358407837bc230a2f98c9fcf2a853fcee7af`;
- private custody manifest: `4edb31f596c8efcc883c14f4f0c67ab31d62348d7501d84757c971cdc97ccd70`;
- complete candidate matrix: `d705967b6e6d635c62cf62b12fde2ba645d63a62785428cf534d921f8d8c6c40`;
- development evaluation: `fd77ffa5ee83c88815eea2cfb577085eb6f704147bfa93f00df20f30dc143641`;
- public development arrays: `bd8fd579da7a9aa7aaf95bad2db41013245643ffa6af44f576b8e7d498c454ed`;
- v15 shadow predictions: `26c48ccd0b23d2b8cd758fcf740811ded013400803e0d20774cddcc880165c77`;
- shadow receipt: `3ed9e7c2df8098efd9936de767cb9796c7284b130fb0b0055295afbf1a62c6d7`;
- no-label packet audit: `aea87ebccd7bf76b233e791ceb3488531c01b3b9f4c9002a8a44fb64894911d3`;
- heldout evaluation: `8b611984c3366b8d154901345de0c6348ef6127069779ea878a7e9a54f6e6f41`;
- heldout evaluation receipt: `8542b7d0a710e70e86f9c25ad35e3e3049217458ed55e37c850c3b1656d6b973`;
- independent final audit: `cf736bb5c68cceaf3d83a7f10e6303307ec6254b22b771610cd379b6551c2f52`.

The first `98784f3` matrix attempt stopped before training because the WSL path parser was malformed. Its output remains a
failed harness artifact and is excluded from every quality claim. Official task data were regenerated after the
`208f968` correction and reproduced the same frozen dataset, release, and custody hashes.

## Next gate

Do not retune v15 or reopen v2 heldout labels. A corrective sorter experiment must use fresh lineages and explicitly
represent the domain shift exposed here: arrays with no unsafe candidate, seed-derived metric intervals, and genuine
`UNRESOLVED` relations among otherwise strong candidates. Architecture identity must remain absent from taste features.
The unchanged v15 checkpoint and the exact deterministic interval policy should both be controls. Confirmation must
again require exact status vectors and exact selected sets on a newly sealed adapter-receipt surface.

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
