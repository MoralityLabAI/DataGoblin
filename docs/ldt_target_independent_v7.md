# Target-independent LDT array gym v7

## Result

V7 removed the policy-lattice circularity found in v6, but no trained
architecture cleared the development or sealed-confirmation gates. The best
sealed pairwise result came from the two-loop h4 model, at 0.8145 accuracy and
0.8370 macro-F1, but it reached only 0.9355 swap consistency, 0.875 ABSTAIN
recall, 0.5833 selected-set exactness, and 0.5000 exact-array accuracy. Its ECE
was 0.1719. There is no retained primary, learned Pareto set, shadow
integration, corpus promotion, deployment, or larger-model authorization.

This is useful negative evidence: acceptable mean ECE did not imply complete
decision behavior or consistent arrays.

## Feature and custody boundary

The eligible `independent` feature view has 45 values:

- primitive left and right interval endpoints;
- interval widths and endpoint-missingness flags;
- separate left and right safety bits; and
- separate normalized parameter-bucket ranks.

It excludes context, fixture-pressure, optimization-pressure, signed deltas,
mismatch flags, dominance/equivalence decisions, rationale indicators, and all
`lattice:*` fields. A unit test replaces the policy function with a failure and
proves that this encoder view does not execute it.

The frozen contract generated 496 labeled development pairs in 24 arrays and
248 target-free confirmation pairs in 12 arrays. Development and confirmation
use eight disjoint lineages and have zero overlap with 28 earlier lineages.
Confirmation pair and array labels were written to private steward custody
before training. Only their hashes appear in the public manifest, and the
capped staging wrapper copied only the public dataset directory.

## Development sweep

Each architecture used seeds 19, 43, and 83 with leave-one-pressure-lineage-out
CV and temperature calibration from out-of-fold development predictions only.
The frozen selector retained one diagnostic seed per architecture and found no
qualified primary.

| Selected diagnostic | Params | CV macro-F1 | Safety | Swap | ABSTAIN | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|
| linear, seed 19 | 405 | 0.4778 | 1.0000 | 0.9355 | 0.0000 | 0.2345 |
| latent h4, seed 19 | 229 | 0.4839 | 1.0000 | 0.9395 | 0.0000 | 0.1469 |
| looped h4 l2, seed 19 | 245 | 0.4822 | 1.0000 | 0.9677 | 0.0000 | 0.1429 |
| looped h8 l3, seed 19 | 513 | 0.4733 | 1.0000 | 0.9274 | 0.0000 | 0.1299 |

ABSTAIN had verifier support in only the controlled-missingness fold. When that
lineage was held out, the training side contained no independent instance of
the missingness-to-abstention mechanism. The zero recall is therefore a real
mechanism-extrapolation failure. It also exposes a dataset-design weakness:
selection-critical classes need support in at least two independent
development lineages if every CV training fold is expected to learn them.

The initial selection receipt encoded zero-support fold recall as -1. Before
the verifier join, it was preserved and superseded by a support-aware receipt
that excludes unsupported folds. Predictions and selected seeds did not
change. The original receipt SHA-256 is
`02a3073ef3ad2e528946a63d428052ef41459ad537462648917ffbb71414fb36`;
the superseding receipt is listed below.

## Single sealed confirmation

The verifier joined the frozen seed-19 predictions exactly once.

| Architecture | Pair acc. | Pair F1 | ABSTAIN | Safety | Swap | ECE | Status F1 | Exact arrays | Selected set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.7984 | 0.8250 | 0.8750 | 1.0000 | 0.9919 | 0.2759 | 0.8218 | 0.5000 | 0.5000 |
| latent h4 | 0.7742 | 0.8058 | 0.8750 | 1.0000 | 0.9435 | 0.1380 | 0.6764 | 0.3333 | 0.5833 |
| looped h4 l2 | 0.8145 | 0.8370 | 0.8750 | 1.0000 | 0.9355 | 0.1719 | 0.7340 | 0.5000 | 0.5833 |
| looped h8 l3 | 0.8024 | 0.8032 | 1.0000 | 1.0000 | 0.8952 | 0.2365 | 0.6340 | 0.3333 | 0.4167 |

Final training saw all development pressures, so controlled-missingness
confirmation was much easier than leave-that-mechanism-out CV: support-aware
pair F1 was 0.9778 for the first three setups and 1.0000 for looped h8. The
weakest confirmation slice was near-tolerance equivalence, where pair accuracy
ranged from 0.4032 to 0.4839. The evaluator records class support and
support-aware macro-F1 for every game and pressure slice so absent classes do
not depress a slice silently.

No setup met the joint safety, orientation, uncertainty, calibration, and
selected-set gates. In particular, the best-ECE architecture was not the best
pair or array sorter.

## Resource receipt

Training ran offline under the verified WSL cgroup contract:

- 2,048 MB RAM maximum and zero swap;
- 50% CPU quota;
- 50 MB/s I/O limit;
- 900-second wall-clock limit; and
- abort rather than uncapped fallback.

The process exited 0. Observed usage was 77.71 MB peak RAM, 72.62 MB average
RAM, 48.15% CPU, and 13.93 MB/s peak I/O. Windows post-run cleanup passed with
no lingering owned or GPU compute processes. No weights, checkpoints, private
labels, or generated runs are committed to the repository.

## Bound receipts

- preregistration:
  `727e311f7a22b0cde37bcb79e5e1bb6697de615b2242d2bde2615e81256259b3`
- public dataset manifest:
  `166667861d31e2a97dd966a167ff7178971c9ba4be33e05553c2ee289b9e74f0`
- private pair labels:
  `149674eeeac125ae50575e801297911e8a64b7b4fba8feae44bdf5ea79656b5a`
- private array labels:
  `3a9367b07e81cecccc1e70994dde6db2f1a0581c2fde3c988d97c4f4469596b1`
- capped run:
  `10a83c506f54ed85be1b03fa82bebdb6c9135ec442a918f9d18f99b3154d9f10`
- trainer summary:
  `83eea7179fc3537c89a967fa6daa0a57ef3453e433645b89d4c36c6320caf3ab`
- cleanup:
  `d94fd8e05e44dd4a091cea9025085811f4d630ca0801c56c2bce1c715a1fb6a6`
- superseding development selection:
  `d6e7b7cbd993f319dc0235a30277f791cb1562cc2e7bb5199f23597a9d9285b8`
- sealed evaluation:
  `76f433949325d2ddb2081786460b4c76b7a9ad86fc306d6d1ed3c46b7c0457dc`

## Next gate

A new development-only contract should cross pressure mechanisms within
lineages so every selection-critical class appears in at least two independent
lineages. It should also test paired swap augmentation or an explicit
antisymmetry loss and an array-consistency objective. Any retained redesign
requires a new sealed confirmation; v7 confirmation is spent and cannot be
used for tuning.
