# Exact status-composition LDT gym v13

## Result

V13 tested whether a learned potential over the complete status composition
could improve on v12's selected-count-only likelihood. It could not. Across
the three frozen seeds, the selected-count control reached 0.6806 mean exact
status-vector and selected-set accuracy. Global composition MAP reached
0.6111, a -0.0694 delta, with per-seed deltas of -0.1250, -0.0417, and
-0.0417. Status macro-F1 also fell in every seed.

The composition decoder itself worked as an exactness mechanism. Against its
matched factorized decoder, global composition MAP raised mean exact recovery
from 0.5000 to 0.6111, a +0.1111 lift. The per-seed exact deltas were +0.1667,
+0.0417, and +0.1250. The two 619-parameter candidates had identical training,
pair behavior, and local status probabilities, while their global decoder
changed 31 target-free confirmation-array predictions. That gain therefore
isolates decoder behavior, but it does not show that the richer composition
objective beats the smaller 416-parameter selected-count control.

The frozen selector retained composition-global seed 127 at 0.6250 exact
status vectors, 0.6250 exact selected sets, 0.5000 worst-fold selected-set
accuracy, and 0.8841 status F1. It missed the 0.6500 exact-status and 0.7500
selected-set gates. The development claim is `not_supported`; confirmation
remains sealed, and no promotion or larger-model training is authorized.

## Model and data contract

The exact composition model enumerates every four-status assignment for arrays
of at most six candidates: at most 4^6 = 4,096 assignments. It learns 210
potentials, one for each `(SELECTED, DOMINATED, UNRESOLVED, UNSAFE)` count
vector whose total is zero through six. Exact marginals provide a proper
likelihood gradient to both the composition potentials and the shared recurrent
status representation. Unit tests compare the partition, status marginals,
and composition marginals against independent brute-force enumeration to
1e-12 tolerance.

Five candidates crossed seeds 43, 79, and 127: a pair reducer, a recurrent
factorized status head, the selected-count global control, and matched
factorized/global composition decoders. The harness was committed and pushed
as `3700078` before official generation.

The official fixture contains six fresh development and four sealed
confirmation lineages across Centauri, They Sing, Court, and Diplomacy. It has
552/350 development/confirmation ordered pairs, 24/16 arrays, and 126/82
tournament reports with array sizes 4, 5, and 6. All 84 v1-v12 lineages were
forbidden. Generation recorded zero prior overlap, zero cross-split overlap,
complete per-lineage critical-class support, and no confirmation targets in
candidate custody.

The whole-array metrics remain bound to the separately implemented v10 metric
audit, which passed 20 of 20 held-out probes across 10 counterexample families.

## Selected diagnostics

The selector retains one seed per architecture using the frozen
worst-fold-first ranking and only then applies every development gate.

| Selected diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact status | Exact selected | Worst selected | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair reducer, seed 43 | 245 | 0.7953 | 1.0000 | 1.0000 | 0.8000 | 0.8035 | 0.4167 | 0.5000 | 0.2500 | 0.0637 |
| recurrent factorized, seed 43 | 409 | 0.7953 | 1.0000 | 1.0000 | 0.8000 | 0.8928 | 0.5417 | 0.5417 | 0.2500 | 0.0637 |
| selected-count global, seed 127 | 416 | 0.7780 | 1.0000 | 1.0000 | 0.8000 | 0.9372 | 0.6667 | 0.6667 | 0.5000 | 0.0452 |
| composition factorized, seed 79 | 619 | 0.7659 | 1.0000 | 1.0000 | 0.8000 | 0.9155 | 0.5417 | 0.5417 | 0.2500 | 0.0663 |
| composition global, seed 127 | 619 | 0.7780 | 1.0000 | 1.0000 | 0.8000 | 0.8841 | 0.6250 | 0.6250 | 0.5000 | 0.0452 |

### Selection-order caveat

Selected-count seed 43 numerically cleared every development threshold:
0.7500 exact status vectors, 0.7500 exact selected sets, 0.5000 worst-fold
selected accuracy, 0.9228 status F1, 0.7953 pair F1, 0.8000 ABSTAIN recall,
and 0.0637 ECE. It was not retained. Seeds 43 and 127 tied on the first two
ranking axes, worst-fold selected and status-vector exactness; seed 127 then
won the preregistered third axis with 0.8375 versus 0.8030 worst-fold status
F1. Seed 127 subsequently failed the 0.7500 mean selected-set gate.

The post-run auditor independently reproduced this rank-then-qualify behavior.
Substituting seed 43 after seeing the matrix would change the frozen selection
rule and is not authorized. The receipt's `no_qualified_primary: true` means
that no *retained* architecture/seed cell qualified; it must not be generalized
to say that no observed cell crossed the numeric thresholds.

## Evaluation review

The eval-review companion manifest declares a deterministic development claim,
no LLM judge, no subjective score, exact candidate and baseline identities,
the lineage firewall, resource and custody checks, and the full decision rule.
The scoped deterministic-development policy returned `warn` with no blocking
finding; its only warning was that the generic pairwise adapter recommends
subjective dimensions, which are explicitly out of scope because exact
environment targets exhaust this claim.

The ordinary base policy returned a release-blocking `fail`: the 24 arrays are
development rather than held out, and there is no subjective A/B display-order
or position-swap protocol. Those are deliberate scope facts, not fields to
falsify for a green report. The base result prevents development receipts from
being presented as release-grade or independently confirmed evidence.

The separate public-receipt auditor had no private-label access. It reproduced
all 15 candidate/seed cells, frozen selection order, gate decisions, bindings,
resource cleanup, matched decoder behavior, and target custody. Its conclusion
layers are:

- metric robustness: `supported_on_declared_independent_v10_holdout`;
- task result: `composition_underperformed_selected_count_control`;
- measurement reliability: `development_receipts_consistent_and_release_scope_blocked`;
- claim support: `not_supported`;
- operational decision: `preserve_confirmation_and_reject_promotion`.

## Resource receipts

Every shard completed offline in its own WSL systemd unit under 2,048 MB RAM,
50% CPU, 50 MB/s I/O, zero swap, and a 900-second wall-clock cap. All 15
PID-scoped cleanup audits passed, with no staged labels, lingering owned
process, external checkpoint, or GPU compute process.

| Candidate family, three seeds | Trainer seconds | Steps | Peak RAM | Peak CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|
| pair reducer | 52.34 | 125,100 | 74.23 MB | 50.18% | 12.182 MB/s |
| recurrent factorized | 113.31 | 254,700 | 33.16 MB | 50.06% | 0.131 MB/s |
| selected-count global | 123.61 | 254,700 | 33.68 MB | 50.05% | 0.131 MB/s |
| composition factorized | 163.83 | 254,700 | 33.80 MB | 50.03% | 0.129 MB/s |
| composition global | 163.16 | 254,700 | 33.72 MB | 50.02% | 0.129 MB/s |

The official matrix completed 1,143,900 steps in 616.25 trainer seconds. The
earlier capped smoke was discovery-only and is excluded from every official
merge, selection, and claim. No generated run, prediction, label, checkpoint,
or model weight is committed.

## Bound receipts

- preregistration: `4c9e709c6c4168b53c29c279769984916cb03b0203c9f77b314dab0d664a6f20`
- eval-review manifest: `dd537a23e801f9ce01c0c03b2f6c819da87fc78d66c685d9c63e80a9a8d80b07`
- deterministic-development policy: `08a055fd70159e142140acbd2c7a62081d1968cb8489352fce9d4144ea6f0a01`
- scoped development preflight: `9cc85cf4664b78bb34436f0a09d57eaecb99ac0df65cc62d4c5e4803967d9ef5`
- base-policy preflight: `5a61138413ef4b7ae9f97e34c43d69bc97f2ac73ef88a32505563763802f219f`
- public dataset manifest: `d5bb1b144b8ecaea566df89821e90b223b4e7c04bec984b266fc2f32fa6e667a`
- generation report: `fdd677b4387dc2b844a1d6592af7e42bab05186d15422c0e93dc48a2b6181cf6`
- private pair-label custody hash: `5e54bf0113eb9554926258f6d4f6197f1310afaf6735e30b5cad6016ffb149f7`
- private array-label custody hash: `7ab6ded6f7a2b4c26c66f73d77e86a3e21206e1e18f7c542d364ae8677925400`
- trainer-plan preflight: `f75bb749180c566c911f72996d3839ffb1a75f9839170c5bb7f12d220df930f0`
- complete shard merge: `fd2de2d7a7cf13b5d32b9e075155083e5a49771743de2685dc1ebd7ab0c8d0dd`
- merged trainer summary: `b6a49ebdff013bb657c3f911fa87050387c496f1252590751fddddc79610cde2`
- merged pair predictions: `825d81547c969cc3a07baed47032a5401eceee3de1f45b5ffb38fec37aa75b79`
- merged array predictions: `79b87b01a1815c9c96059b78173c35e7002e619a050299d4b1744f42d390187c`
- development selection: `7e88eb13f32cfb85008385945f80930b918a60372fb3f596882afd2956602e50`
- selected trainer summary: `8b7c69ec46caf74236dda47988cd6ec5939daded987071b577b67c5a23fb3b88`
- selected pair predictions: `c86bd6c5b64ba05ff97ef563f6881fd0dcf1588cb1b314f988817da56634cfd1`
- selected array predictions: `e74e004e7fdc37de7db22171c0795d2de8f0b0829a34cef3f7b635c365a9ed7e`
- public-receipt auditor: `450e8ae5f78d59a4c87e42d556de3db11d8e8f16e879ec18f15c5e0306f4d0bf`
- post-run public-receipt audit: `a31c25413cada5f50e5a16adc4c7ee62166bd4fb20a62decbcd5ce0e95275997`

## Next gate

Do not tune on v13 lineages or promote selected-count seed 43 post hoc. Before
another architecture search, freeze fresh lineages and make seed-selection
semantics explicit. A gate-first contract should filter seed cells through all
hard thresholds before applying the diagnostic ranking, while preserving a
separate rank-only diagnostic when no cell qualifies. Retain the smaller exact
selected-count global model as the control; v13 provides evidence against
spending more parameters on unconstrained all-status composition potentials.

LoRA or HRM-Text scale remains downstream of a fresh development-qualified
sorter and one independent held-out confirmation pass.
