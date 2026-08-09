# Cardinality-decoder LDT gym v11

## Result

V11 isolated a learned selected-count decoder after v10's edge and soft-Dice
heads failed the exact-set gate. Two 451-parameter candidates trained identical
candidate-local status heads and identical stop-gradient cardinality readouts.
Their only difference was inference: factorized per-candidate argmax versus
joint MAP over the learned selected count.

Across the three frozen seeds, joint MAP raised mean exact status-vector and
selected-set accuracy from 0.4583 to 0.5278, a 0.0694 delta. The effect was not
robust: per-seed exact deltas were +0.2083, +0.0417, and -0.0417. Mean status
macro-F1 fell from 0.8836 to 0.8650, and worst-fold exactness remained 0.2500
for every candidate and seed.

No candidate cleared the preregistered 0.6500 mean exact-status, 0.7500 mean
exact-selected-set, and 0.5000 worst-fold selected-set gates. The selector
recorded `no_qualified_primary: true` and
`confirmation_join_authorized: false`. Private pair and array labels were not
staged, resolved, or read; the confirmation evaluator was not run, and the
one-reveal budget remains unspent.

There is no retained primary, checkpoint or corpus promotion, deployment
claim, or larger-model authorization.

## Frozen design and custody

The preregistration was committed and pushed before the official fixture was
generated. The fixture has six fresh development lineages and four fresh
sealed confirmation lineages across Centauri, They Sing, Court, and Diplomacy.
It contains 552/350 development/confirmation ordered pairs, 24/16 arrays, and
126/82 tournament reports with array sizes 4, 5, and 6.

All 64 v1-v10 lineages were forbidden. Generation recorded zero prior overlap,
zero cross-split overlap, disjoint contexts, complete critical-class support in
every lineage fold, and both orientations for every unordered pair.
Confirmation inputs contain no targets. The two exact metrics remain bound to
the separately coded, environment-owned v10 robustness audit that passed all
20 held-out probes across 10 counterexample families.

The cardinality head reads the pooled four-hidden-unit recurrent state plus
normalized array size and predicts selected counts 0 through 6. It is trained
with frozen weight 0.5 categorical cross-entropy after each status update,
without sending count gradients into the shared status representation. Joint
MAP scores every permitted selected count using that posterior and local
selected-versus-best-nonselected log odds. It does not use game, pressure,
policy decisions, hidden targets, or verifier execution.

## Selected diagnostics

The frozen worst-fold-first selector chose seed 37 for the pair reducer and
seed 67 for the other three architectures.

| Diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact status | Exact selected | Worst exact selected | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair reducer | 245 | 0.8120 | 1.0000 | 1.0000 | 0.8000 | 0.8514 | 0.5000 | 0.5000 | 0.2500 | 0.0789 |
| recurrent factorized | 409 | 0.7751 | 1.0000 | 1.0000 | 0.8000 | 0.8879 | 0.4583 | 0.4583 | 0.2500 | 0.0731 |
| count readout, factorized | 451 | 0.7751 | 1.0000 | 1.0000 | 0.8000 | 0.8879 | 0.4583 | 0.4583 | 0.2500 | 0.0731 |
| count readout, joint MAP | 451 | 0.7751 | 1.0000 | 1.0000 | 0.8000 | 0.8540 | 0.5000 | 0.5000 | 0.2500 | 0.0731 |

The recurrent and ignored-readout controls produced identical development
metrics and byte-identical target-free status probabilities and predictions.
The two cardinality candidates also produced identical local status
probabilities; joint decoding alone changed the assignments. This verifies the
intended isolation. The evidence supports only a mixed-seed directional exact
lift, not a reliable decoder improvement.

Calibration, safety, and orientation are again not the bottleneck. Exact-set
errors remain concentrated enough that every lineage-heldout validation regime
contains a 0.2500 fold.

## Post-run evidence review

A separate public-receipt auditor checked frozen preregistration and dataset
bindings, complete candidate/seed coverage, all cap and cleanup receipts,
target custody, matched pair behavior, ignored-readout identity, identical
local probabilities for the decoder ablation, and an actual change in decoded
assignments. It had no access to private confirmation labels and has no
promotion authority.

The audit judgment is `whole_array_gate: not_supported`,
`decoder_mechanism: directional_mixed_seed_only`, and
`operational_action: preserve_confirmation_and_reject_promotion`.

## Resource and recovery receipts

All 12 authoritative candidate/seed shards completed with zero exit codes
under isolated caps of 2,048 MB RAM, zero swap, 50% CPU, 50 MB/s I/O, and 900
seconds. Every PID-scoped cleanup audit passed and no private labels were
staged.

| Candidate family, three seeds | Trainer seconds | Steps | Peak RAM | Peak CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|
| pair reducer | 50.24 | 125,100 | 36.39 MB | 50.08% | 1.765 MB/s |
| recurrent factorized | 112.48 | 254,700 | 33.05 MB | 50.05% | 0.119 MB/s |
| count readout, factorized | 143.16 | 254,700 | 32.78 MB | 50.04% | 0.117 MB/s |
| count readout, joint MAP | 159.73 | 254,700 | 33.20 MB | 50.01% | 0.117 MB/s |

The authoritative matrix completed 889,200 steps in 465.61 trainer seconds.
Two earlier launcher attempts are discarded operational failures: the first
was interrupted by the orchestration timeout before producing a run receipt;
the second encountered that attempt's transient unit and aborted before
training. The unit was cleared, and neither partial root entered the complete
matrix, merge, or selection.

No generated run, prediction, checkpoint, weight, private label, or hidden seed
is committed to the repository.

## Bound receipts

- preregistration: `4e861ed0c1180a8606f3ba5b7fcc211edfb145304ebdc4b1b190bdd86434a85c`
- public dataset manifest: `a226340cba00dba7898d07a18c90f11b78b207b651b4086f75e5bff5326f596b`
- generation report: `2e5e83b4d0e762c80c175b7f63d7ee16da6566e7d65c0b03d0c808e4383fa12c`
- private pair-label custody hash: `4016c05cbe7081511bb7bee21eecca2ef1a8e13750085ae852a42cdd5987f879`
- private array-label custody hash: `6a338187cd6b06d70dcd31191415a1d7bade616f1581912335f6a99ee4e9a8de`
- trainer-plan preflight: `d4e0d66168696165eaa83bb4d0bc3bc99330531386e31d0ecd437b52c9c78b2e`
- complete shard merge: `bc130235c24c4474badaa912932e614203728b871bc53dc1eed67690dd87aa6f`
- merged trainer summary: `44d68e39df90d3475206221babf98282be160d3c81ad333086be21a8fddc205f`
- merged pair predictions: `2269fad306e8e6d30e930ff9f18cb158a1f70f435330fda987a2f9aefa73aa98`
- merged array predictions: `2a4285c0ef478e2ecce973c88f3d51d55229d2d3552789b16c94cc7078120a7f`
- development selection: `9e94e3208b5017f66583cd2bb1083c5b8facbf9506e4eb77c27c21246e290979`
- selected trainer summary: `5a82398a87219e8122deecf9dab928be63fb21cd6b9087964d5dd74742546eb0`
- selected pair predictions: `a6607406b8652f17cf667e993f2e709662fb2ceed036ff5436031976b27201f9`
- selected array predictions: `f9ae221a0dfea90643c0e2e48eb8ba60c24f30b8d13a9ddaea63e5aadcd6017c`
- post-run evidence audit: `827963afcb9e6310a4a6844b6e6ea9127090be65daf52f14d88d22ccd3bff83c`

## Next gate

Do not tune cardinality weights or decoders on these lineages. A future,
separately frozen small-model contract should compare factorized status
cross-entropy with a globally normalized full-array objective. Arrays of at
most six candidates permit exact enumeration of all status vectors, enabling
an exact structured likelihood rather than a detached count heuristic. The
model must remain target-independent at inference, use fresh lineages and a
seed ensemble, and preserve the same mean and worst-fold gates. LoRA or HRM
scale remains downstream of one development-qualified sorter and one
independent confirmation pass.
