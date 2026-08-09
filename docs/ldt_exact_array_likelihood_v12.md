# Exact-array likelihood LDT gym v12

## Result

V12 replaced v11's detached count readout with a proper, globally normalized
status-vector likelihood. It trains the recurrent status representation and a
selected-count potential together using exact structured marginals. Two
416-parameter candidates then use identical learned probabilities and differ
only at inference: factorized argmax versus global MAP.

Global MAP improved exact status-vector and selected-set accuracy in all three
frozen seeds by +0.1667, +0.1250, and +0.0833. The three-seed mean rose from
0.4722 to 0.5972, a +0.1250 matched-decoder lift. Worst-fold selected-set
accuracy improved by +0.2500 in two seeds and tied in one. Mean status F1 was
nearly neutral, with per-seed deltas +0.0152, -0.0341, and +0.0061.

This is the first consistent multi-seed exactness lift in the array succession,
but it is still not a pass. The frozen selector retained global-MAP seed 41 at
0.6250 exact status vectors and selected sets, 0.5000 worst-fold selected-set
accuracy, and 0.9076 status F1. It missed the 0.6500 mean exact-status gate and
the 0.7500 mean selected-set gate. Seed 113 reached 0.6667 exactness but still
missed 0.7500 and was not selected under the preregistered worst-fold-first
ranking.

No candidate qualified. The selector recorded `no_qualified_primary: true`
and `confirmation_join_authorized: false`. Private pair and array labels were
not staged, resolved, or read; no confirmation evaluator ran, and the
one-reveal budget remains unspent. There is no retained primary, checkpoint or
corpus promotion, deployment claim, or larger-model authorization.

## Frozen design and exactness check

The preregistration and implementation were committed and pushed before the
official fixture was generated. The fixture contains six fresh development
lineages and four fresh sealed confirmation lineages across Centauri, They
Sing, Court, and Diplomacy: 552/350 development/confirmation ordered pairs,
24/16 arrays, and 126/82 tournament reports with sizes 4, 5, and 6.

All 74 v1-v11 lineages were forbidden. Generation recorded zero prior and
cross-split overlap, disjoint contexts, complete critical-class support in
every lineage fold, and complete pair orientation. Confirmation inputs contain
no targets. The exact status-vector and selected-set metrics remain bound to
the independently coded environment-owned robustness audit that passed 20 of
20 held-out probes across 10 counterexample families.

For an array, the model assigns each full status vector the sum of local
log-probabilities plus a learned selected-count potential, then normalizes over
all four-status assignments. Since the global potential depends only on the
selected subset, exact partition, count marginals, and candidate marginals are
computed over at most 64 subsets while analytically summing nonselected labels.
Unit tests compare those values against brute-force enumeration of every
status vector to 1e-12 tolerance.

The objective uses no class reweighting because it is a proper likelihood. Its
structured marginal-minus-target gradient updates the shared recurrent state;
the selected-count potential is trained in the same array step. Inference uses
only the 23 target-independent candidate primitives. It receives no game,
pressure, verifier policy, hidden target, or confirmation label.

## Selected diagnostics

The frozen selector chose seed 41 except for exact-likelihood factorized, where
it chose seed 113.

| Diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact status | Exact selected | Worst exact selected | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair reducer, seed 41 | 245 | 0.8003 | 1.0000 | 1.0000 | 0.8000 | 0.8292 | 0.4583 | 0.4583 | 0.2500 | 0.0592 |
| recurrent factorized, seed 41 | 409 | 0.8003 | 1.0000 | 1.0000 | 0.8000 | 0.8877 | 0.5417 | 0.5417 | 0.2500 | 0.0592 |
| exact likelihood + factorized, seed 113 | 416 | 0.7931 | 1.0000 | 1.0000 | 0.8000 | 0.9070 | 0.5833 | 0.5833 | 0.2500 | 0.0504 |
| exact likelihood + global MAP, seed 41 | 416 | 0.8003 | 1.0000 | 1.0000 | 0.8000 | 0.9076 | 0.6250 | 0.6250 | 0.5000 | 0.0592 |

The exact objective without global decoding only slightly improved the
three-seed mean over the recurrent factorized objective and was mixed by seed,
so objective benefit is not isolated here. In contrast, the decoder ablation
has identical training, parameter count, pair behavior, and local status
probabilities. The exactness improvement is therefore attributable to global
decoding on the declared development surface.

## Post-run evidence review

A separate public-receipt auditor verified frozen bindings, 12-cell coverage,
all cap and cleanup receipts, lineage and target custody, matched objectives,
identical pair behavior and local probabilities, changed global assignments,
and a positive exactness delta in every seed. It had no private-label access
and has no confirmation or promotion authority.

Its judgment is `whole_array_gate: not_supported`,
`global_decoder_exactness: consistent_lift_on_declared_development_only`, and
`operational_action: preserve_confirmation_and_reject_promotion`.

## Resource receipts

All 12 candidate/seed shards completed with zero exit codes under isolated
caps of 2,048 MB RAM, zero swap, 50% CPU, 50 MB/s I/O, and 900 seconds. Every
PID-scoped cleanup audit passed and no private labels were staged.

| Candidate family, three seeds | Trainer seconds | Steps | Peak RAM | Peak CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|
| pair reducer | 54.03 | 125,100 | 74.77 MB | 50.16% | 10.693 MB/s |
| recurrent factorized | 112.71 | 254,700 | 33.56 MB | 50.05% | 0.125 MB/s |
| exact likelihood, factorized | 122.17 | 254,700 | 33.30 MB | 50.06% | 0.125 MB/s |
| exact likelihood, global MAP | 127.48 | 254,700 | 33.38 MB | 50.02% | 0.123 MB/s |

The complete matrix performed 889,200 steps in 416.39 trainer seconds. A
separate precommit capped smoke was discovery-only and excluded from all
official merge, selection, and claims.

No generated run, prediction, checkpoint, weight, private label, or hidden seed
is committed to the repository.

## Bound receipts

- preregistration: `d42719b40204543587f48f9310d152fc3d68c4ed2f9ceba32e6e9b83dc635409`
- public dataset manifest: `e651d83289c490f21f99f46da890e7c501638b9dd0a07c20a224e91d59246fd3`
- generation report: `97109efd0f6bf43a8c0bb17b7f793a3fe6f3e95949d5067b54664ef0677e1e13`
- private pair-label custody hash: `764c37c6510ed224a1abfdffe67b48d984bd585d3407fce718f6093938db1657`
- private array-label custody hash: `b6734c3eaf3baeb85794dc6cb1a7e5e19a3db09b428220793dcc6107f7a0b151`
- trainer-plan preflight: `434bda641aff35410e2fcb62842f9c2e42b74274d2db9eea00b66ea23db378d3`
- complete shard merge: `ba606f8ea359d8e7e9b4aec39465c35d71a8ecd32825a402a37fa63055fc1864`
- merged trainer summary: `2811a7ff977d85c44a6b382f530ca26f38d5eab752871fc7fb8f9d1addaafbbb`
- merged pair predictions: `3b116a23fae395de5a4094a79b33e9f14b06e4531c7aebc61c77fd801bf0a387`
- merged array predictions: `416e4959b48740e74ef02fc04f3b43c34a6faf4864ec05f0033064c2605fcaa9`
- development selection: `62a92057c8ecc5fb5864dcfad8e03b6e1ff7052c08e4d0da0e7a0cfa23e3cb33`
- selected trainer summary: `87b5e2d9697ce917e14c03b6777d253a408dee2d4a2faec6f1e0a9712ed1f73f`
- selected pair predictions: `d02bbeb1b8b4aeeb691f19488b367e87d3df10daabc826813d02c4da5e004c10`
- selected array predictions: `0687650beab016e7b81ca8c461d9a1ac9b32a6abeb98378c75e7ab73975079d7`
- post-run evidence audit: `c2670eebed9c455c8d37a074e3cd480dbe8d2bb8b1fb6b869a17b175513b7107`

## Next gate

Do not tune v12 on these lineages or select seed 113 post hoc. A future frozen
small-model contract should retain exact global training and decoding while
replacing the single selected-count potential with a richer all-status
composition or symmetric compatibility potential. That tests whether the
remaining complete-vector errors arise from unresolved/dominated/unsafe count
structure rather than selected cardinality alone. It must use fresh lineages,
the same seed and worst-fold discipline, and the sealed confirmation firewall.
LoRA or HRM scale remains downstream of a development-qualified sorter and one
independent confirmation pass.
