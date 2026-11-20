# DataGoblin module lab for the original 1B pretrain

The module lab supplies new checked data, matched module comparisons, and
an end-to-end dry run for the planned original pretrain. The parameter
allocation in the Organelle Gym design is an envelope to test with actual
instantiated weights. The local comparison uses a 256 MiB process-tree cap.

## Candidate coverage and checked data

The preceding Lean ranker found all 28 goals for which its candidate pool
contained a verified proof. The other 32 goals motivate the next comparison.
The Lean pool lab compares three proposal orders on the same fresh goals.
The first uses the previous candidate generator. The second retrieves more
role-relevant declarations and pairs or triples them with the old tactic
forms. The third adds explicit rewrite chains and controlled simplification.

The comparison freezes a training cohort and a separate structural holdout.
Native tactics run first on the holdout. The held-out analysis reports
candidate coverage on the remaining stalls, including retrieval of each
independently replayed witness's lemmas. It evaluates prefixes at 6, 12, 24
and up to 36 candidates. Every proposed tactic runs from the exact goal
through the strict LDT gate. A shared replay caches each distinct checked
action per goal. Report both actual deduplicated work and each policy's
first-certificate prefix costs.

    python scripts/run_rlt_lean_ldt_windows.py evaluate --composition --out RUN --previous-fixture PREVIOUS_FIXTURES --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package LCGRT_ROOT
    python scripts/run_rlt_lean_ldt_windows.py certify --composition --out RUN --ramcap-package LCGRT_ROOT
    python -m datagoblin.training_gym.rlt_lean_pool_export RUN CHECKED_PACK

The export identifies goals by theorem content and lineages by construction
template. The public files carry premises, environment-bound epoch,
retrieved lemma names and ordered actions. Separate files retain verified,
failed and inconclusive outcomes plus verifier work. Training consumes
only the training split. Verified labels bind to the goal, candidate
and verifier identity.

## Matched module experiment

Share a small operator between the reasoning and answer phases. The
reasoning phase advances a task-specific message, such as a graph edge or
retrieved-lemma context. The answer phase updates a continuous proposal
state. Register five arms with the same parameter count, input adapters,
batches, readouts, and planned block calls. Compare direct feedback with
RLT and coefficient-matched damping at either the reasoning boundary
or the answer boundary. Repeat with matched seeds and compare checkpoint
training schedules separately from runtime swaps.

Use Lean candidate pools and rule-revision episodes as the first two
typed lanes. Freeze a text lane to exercise the shared pretraining path.
Train on token loss for the text lane, verified candidate set likelihood
for Lean, and the before/after endpoint objective for rule revisions.
Log raw input tokens, supervised target tokens and objective weights by
lane. Run held-out evaluation with immutable candidates, a shared
lexical or native control, and a separate LDT verifier. Use a current
evidence epoch after a rule edit and record verifier work alongside
neural visits. Measure cold-start and persisted-state behavior.

The mixed pilot's input spec names three local directories through
`checked_lean_pack`, `text_fixture` and `text_holdout` keys. Point the
first at the expanded Lean export. The other two point at the frozen
byte-model training fixture and the replication holdout. Preparation
verifies all input hashes, streams a bounded text sample, and generates
a fresh revision set. Optimizer batches draw from each lane within the
same update. All five placements share each seed's minibatch plan.

    python scripts/run_rlt_lean_ldt_windows.py prepare --pretrain --out MIXED_RUN --input-spec INPUTS.json --ramcap-package LCGRT_ROOT
    python scripts/run_rlt_lean_ldt_windows.py fit --pretrain --out MIXED_RUN --ramcap-package LCGRT_ROOT
    python scripts/run_rlt_lean_ldt_windows.py evaluate --pretrain --out MIXED_RUN --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package LCGRT_ROOT

Training and scoring use the optional pilot dependencies. The evaluator
separates Lean replay of checked outcomes from live graph verification.
It records the realized block visits, underlying Lean proof work, and
edge traversals. It scores matched checkpoints at depths 4, 6 and 10,
and evaluates all schedule swaps at the training depth of 6.

Two direct-feedback controls measure the data recipe separately. The
text-first curriculum shifts loss emphasis across thirds of the same
update sequence. Its examples and average loss weights match the
constant mixture. The text-heavier arm exchanges two Lean pools for
two text windows per step. It holds the graph draws and total example
cell calls fixed, with costs and target exposures reported by lane.
Both controls use the same initial weights as the direct arm.

The [Campsite repair lane](rlt_hermes_repair.md#training-pack) supplies a checked repair objective for the answer-state comparison. Fit direct, RLT and coefficient-matched damping on shared training targets with fixed planned visits. Verify decoded grids at the final visit and at each possible exit. Record repair coverage, state and update RMS, block calls, and verifier calls. The 1B [mixture plan](../training/data/aleph_20b_30b_mix.json) assigns the task to the procedural-games lane.

## Integrated dry run and pretrain recipe

The integration check should exercise data loading, training, checkpoint
resume, and verified evaluation in one run. Account for unique weights
and execution count separately. Show which shared parameters receive
gradients from each loss, where interpolation runs, and how the router
advances or resets each state. Keep independent evaluation data outside
the optimizer and selection loop.

Use the measured task and transfer results to set a starting mixture,
curriculum stages, promotion comparisons and stop conditions for the
1B recipe. Include throughput and memory measurements with their hardware
scope, then specify how the full-size run must be profiled before its
training budget is registered. [Current routing evidence](rlt_routing_decision.md)

The design budget has a separate exact parameter ledger. For one
planning layout it counts 999,352,320 unique weights with tied embeddings,
bias-free grouped-query attention, SwiGLU and RMS norms. At six reasoning
and six answer visits the layout executes 74 Transformer blocks. The routing
policy sets the executed block count. Choose actual visits and retained
modules using mixed-pilot results.
The memory ledger shows its assumptions for weights, gradients, masters
and Adam moments. Measure activations, sharding, kernels and throughput
on the intended training hardware before dispatching a scale run.

    python -m datagoblin.training_gym.rlt_pretrain_scale --reasoning-visits 6 --answer-visits 6

Create the local replay bundle after the certificate and mixed evaluation
phases have completed and the recipe has been written.

    python -m datagoblin.training_gym.rlt_pretrain_export RUN CHECKED_PACK MIXED_RUN RECIPE ARCHIVE

The archive includes the frozen input rows, checked outcomes, benchmark
traces, small fitted checkpoints, and a SHA-256 manifest. Extract into a
scratch directory for replay. The `mixed` directory uses the runner's
on-disk layout. To rescore all arms, point the evaluation command at a
scratch copy with its `eval` directory removed and use the registered
source and verifier. The Lean `certificates` files can also be replayed
with the registered Lean toolchain.
