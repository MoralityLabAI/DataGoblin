# Structured LDT array gym v8

## Result

V8 fixed the development-support and orientation defects exposed by v7, but
no candidate cleared the frozen development gate. Exact antisymmetric
projection made swap consistency 1.0000 by construction, and the structured
pair weighting raised mean array-status macro-F1 as high as 0.8711. Mean
selected-set exactness nevertheless reached only 0.5625, below the
preregistered 0.7500 requirement.

The development-only selector therefore recorded
`confirmation_join_authorized: false`. Private pair and array labels were not
read, the evaluator was not run, and the one-reveal confirmation budget
remains unspent. There is no retained primary, corpus or checkpoint promotion,
deployment, shadow integration, or larger-model authorization.

## Dataset and custody

The frozen fixture contains four development and four confirmation lineages.
Every lineage crosses all four mechanisms:

- near-tolerance dominance;
- near-tolerance equivalence;
- controlled missingness; and
- boundary Pareto trade-off.

Every critical preference, rationale, and array status occurs in all four
development lineages. Consequently, each leave-one-lineage-out training and
validation fold has positive support for every critical class. The fixture has
350 ordered pairs, 16 arrays, and 82 reports in each split. Array sizes 4, 5,
and 6 are present; all unordered pairs have both orientations.

The candidate feature view remains the 45-value target-independent primitive
view introduced in v7. Context, pressure, policy decisions, rationale
indicators, and lattice fields are excluded. Development and confirmation
contexts are disjoint, the eight lineages are fresh relative to the 36
forbidden earlier lineages, and the feature audit found no forbidden proxy.

Confirmation inputs are target-free. The environment-owned pair and array
label files were sealed before training outside the staged public dataset.
Only their hashes and custody receipt appear in the public manifest.

## Objective ablation

Five setups used seeds 23, 47, and 89. Selection retained one diagnostic seed
per architecture using the frozen worst-fold-first ranking. Calibration used
only out-of-fold development predictions.

| Selected diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact arrays | Selected set | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| looped h4 pair baseline, seed 89 | 245 | 0.7993 | 1.0000 | 0.8927 | 0.8000 | 0.6563 | 0.3750 | 0.5625 | 0.0564 |
| looped h4 antisymmetric, seed 89 | 245 | 0.8081 | 1.0000 | 1.0000 | 0.8500 | 0.8320 | 0.5625 | 0.5625 | 0.0810 |
| looped h4 structured, seed 89 | 245 | 0.8083 | 1.0000 | 1.0000 | 0.8000 | 0.8711 | 0.5000 | 0.5000 | 0.0701 |
| latent h4 structured, seed 47 | 229 | 0.8055 | 1.0000 | 1.0000 | 0.8000 | 0.8498 | 0.4375 | 0.4375 | 0.0643 |
| linear structured, seed 89 | 405 | 0.8038 | 1.0000 | 1.0000 | 0.8000 | 0.8464 | 0.5625 | 0.5625 | 0.0591 |

The paired-orientation batches keep each forward/reverse pair in the same
minibatch. Antisymmetric inference averages a forward preference probability
with the inverse-label probability of its swapped input, then emits the exact
inverse for the swapped decision. The structured variants additionally weight
public development edges associated with unsafe, dominated, unresolved, and
selected trade-off statuses by 1.75.

Across all 15 runs, OOF ECE ranged from 0.0396 to 0.0925. Calibration, safety,
and class support are no longer the limiting factors. The remaining failure is
listwise: improved pair and per-item status predictions do not yet recover the
complete selected set reliably.

## Confirmation boundary

The selector wrote one frozen seed and target-free prediction set per
architecture before any possible verifier join. None qualified because every
selected seed missed mean selected-set exactness of 0.7500. The private
confirmation files were not opened or joined, so there are deliberately no v8
held-out performance numbers in this report.

This preserves the distinction between a development claim and independent
verification. The unspent confirmation cannot be used for architecture
tuning; it may be revealed once only after a future, separately frozen
development contract produces a qualified primary.

## Resource receipt

Training ran offline under the verified WSL cgroup contract:

- 2,048 MB RAM maximum and zero swap;
- 50% CPU quota;
- 50 MB/s I/O limit;
- 900-second wall-clock limit; and
- abort rather than uncapped fallback.

The process exited 0 after 304.45 seconds and 521,550 bounded training steps.
Observed usage was 80.57 MB peak RAM, 71.09 MB average RAM, 45.33% CPU, and
14.58 MB/s peak I/O. Windows post-run cleanup passed with no lingering owned
or GPU compute process. No weights, checkpoints, private labels, predictions,
or generated runs are committed to the repository.

## Bound receipts

- preregistration:
  `6e2eed040c4b1f8d9f87823e09ca6c2c51f165bf6b8c2656a3616ccc353e547b`
- public dataset manifest:
  `d67da2107a92dc194dc33c10464566c54199f8a655b40b4d6e3484f4d6f6c798`
- private pair labels, hash disclosed by public custody receipt:
  `154c2f8c859eafc114edee96cf32e0caa29ca22a89d7f47408f80ba69e964ed8`
- private array labels, hash disclosed by public custody receipt:
  `18fa582b39cc5195ebf45a08ce20fa38c18f8589a2da863f140dd7b916cdc70a`
- capped run:
  `4e9dffbb4d4ac7cea72edee69af6f0aba41f4b12697d20805ec02e8461fc5a51`
- trainer summary:
  `e660921b60184907180f6cb0be3122e556b19c33d5345b16e6c36317624e4d08`
- target-free predictions:
  `071bbafa595e2729645c6e933841bebd10b74c4834668c02a65b1144a17ae0f6`
- cleanup:
  `c47a1a4e749ea477815585da44ba9ddc7c90cd8ab5569a59b4b9540e110cb90a`
- development selection:
  `c8c5617bfb5b49d33bd74598af5fc3c2cc0514994ed4798df4272e164dc756e5`

## Next gate

A new development-only contract should test an actual listwise objective over
the reducer's selected set and full status vector, rather than infer array
quality only through weighted pair losses. It should use more mixed-mechanism
development lineages so exact-set estimates have finer resolution, retain the
exact antisymmetry construction, and compare pooled node/edge message passing
against the current linear, latent, and looped controls.

The fresh contract must still select without confirmation labels. Only a
development-qualified primary may spend the remaining sealed reveal.
