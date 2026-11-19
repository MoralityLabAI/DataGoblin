# RLT research track

Start with a shared-loop Transformer and ask when the remaining-time update improves a task
at the same compute cost. Use HRM-Text-1B later as a frozen host and external comparator.
This track can measure the value of an RLT organelle inside HRM. A decision about retiring HRM
requires a separate checkpoint-scale comparison.

## What to reproduce

For a budget of K visits, the shared block proposes an endpoint h at visit k. RLT advances
the carried state x by `(h - x) / (K - k)`, with k starting at zero. The first update takes
1/K of the proposed displacement and the last takes the whole displacement. A constant
endpoint yields evenly spaced states. A moving endpoint changes the path, and an expanding
block can still amplify perturbations. The paper's guarantees depend on regularity of the
block. Its model also conditions on time and step size, supervises every block output and
decodes the final output. [Paper, sections 3-4 and appendices C-F](https://arxiv.org/html/2611.16265v2)

| Comparison | Evidence to collect |
|---|---|
| Published reference | The paper's 278M model reports lower Pile, FineWeb-Edu and OpenWebText perplexity at each matched depth. Ten-task mean accuracy rises at K=4 (42.38% to 43.30%) and falls at K=8 (44.36% to 43.53%). At K=32, the K=8-trained RLT retains much more performance than direct feedback in that experiment. |
| First known benchmarks | Freeze text windows for FineWeb-Edu and OpenWebText. Score LAMBADA, HellaSwag and both ARC splits with the paper's scoring convention. Preserve task-level results. Use exact released checkpoints and eval items if available. Otherwise label the results a reduced-scale replication with its own pinned data and recipe. |
| Mechanism controls | Train direct, RLT and constant damping from matched initializations with the same examples, token budget, recurrent block calls, time inputs and readout losses. Match damping's coefficient sum and last update to RLT. Report actual cumulative displacement too, since the coefficient sum does not match realized displacement or downstream compute. |
| Weight versus schedule | At inference, cross the checkpoint's training rule with the runtime rule. Report these cells separately from native inference. Sweep K=2,4,8,16,32 for K=8 training and start an independent trajectory for each K. A visit from the middle of a K=32 trajectory and the last visit of a short run have different time inputs and coefficients. |
| Scale versus direction | Measure the carried state and endpoint separately before normalization. Track residual RMS, realized update RMS, logit change, and perturbation gain before and after feedback. Pair perturbations by item, visit and random direction. Compare fixed absolute and relative doses and include task scores to distinguish a scale change from a useful computation. |

Our earlier small Lean runs found lower raw state growth, often matched by damping. Their
completion counts need replay through the repaired verifier. The interrupted grounded Lean run
trained direct-feedback rankers and did not produce a fresh RLT result. Use the clean Lean data
as a later transfer test with native automation and the identical-pool heuristic as controls,
at matched kernel calls and wall time.

## First novel evaluation

The rule-revision pack isolates task horizon and response to new evidence. Within each lineage,
two episodes share the same 32-link graph, initial vertex and hop count. One replaces a
single link on the queried path and the other reissues its existing destination. The model must
commit its first answer before it sees the update, then answer the same query under the new rules.
Compare fresh context with declared persistent state. For persistent execution, tag the correction
with a new evidence epoch and log whether the system invalidated stale state.

| Split | Hops | Edit position |
|---|---:|---|
| Training and development | 3-5 | Interior |
| Length holdout | 7-9 | Interior |
| Revision-position holdout | 3-5 | First or last |

All variants of a graph and start vertex stay in one split. Holdouts probe length and edit position
within this generator. Train and evaluate over distinct graph lineages with common formatting and
32 links at each horizon. Use development to pick difficulty and stop rules. Cluster intervals by
lineage instead of treating the two variants or multiple K values as independent examples.

Pass ordered `links`, `start` and `steps` to the first turn and ask for the endpoint after
exactly that many hops. Record the answer before showing `patch.source` and `patch.destination`.
On the second turn, replace only that link and ask again from the original start. Keep evaluation
targets with the scorer. Export one prediction object per item with `id`, `before` and `after`.
The scorer checks file hashes and exact ID coverage. It reports per-turn and joint success,
correctness on both variants, and stale answers when the original answer was correct. Its counts
are `[hits, denominator]`. The scorer sees final predictions, so the model runner must enforce
turn order and retain its turn-by-turn trace.

    python -m datagoblin.training_gym.rlt_revision build --out runs/rlt-fixtures --seed 17
    python -m datagoblin.training_gym.rlt_revision score --pack runs/rlt-fixtures --split development --predictions run.jsonl

Seed 17 is a public smoke example. Generate fresh packs with steward-held seeds and move
evaluation targets outside the model's workspace before a real assessment. The runner should
expose only the training partition for updates. Later families can add composed rules and a
separate source-clean Lean transfer test.

## Hybrid experiments

| System | Minimal causal comparison |
|---|---|
| TRM refinement loop | Apply remaining-time interpolation to one continuous refinement state. Match direct and damping at the same host, bridge, TRM invocations and readout. Keep the discrete answer and halt decision as separate outputs. |
| Typed checker | Add a checker at a named proposal boundary. Compare no checker, checker alone, direct plus checker and RLT plus checker. Interpolate latent state only. Apply a typed update only after verification with the current evidence epoch. Track precision, coverage, rejections, and fallback time. |
| Frozen HRM-Text-1B | After mask, scoring and replay fidelity checks, add a bounded residual organelle at a single named boundary. Compare native host, identity bridge, direct, damping and RLT with the same intervention capacity, dose, execution count and gradient access. Try independently trained module replacement only after this test. |

## DataGoblin evidence

Use paired rows as experiment evidence. Record parent checkpoint and source hashes, item and
lineage, training rule, runtime rule, model and seed, K at train and test, block and checker calls,
actual memory and time, token counts, final decisions and bounded per-visit summaries. Attach the
matched control and both outcomes, including regressions and aborted runs. Keep calibration,
curator reward, development and final reporting separate. Before training a curator on
architecture outcomes, repeat promising contrasts from shared checkpoint and optimizer snapshots.
Freeze promotion criteria before scoring the final split. The fixture CLI produces deterministic
task and checker evidence. It makes no architecture-promotion decision and requires no checkpoint.

## Small matched workstation pilot

The Windows pilot uses one worker and a hard 256 MiB Job Object commit limit. It admits a run
only when that allowance fits above 3 GiB of available host memory, then stops if available
memory drops below the reserve. It uses the existing audited `lcgrt.ramcap` helper from the
LoopedTransformers checkout. The model is a 32-wide, two-head causal Transformer block with
tied readout, time and step-size inputs, and four training visits. NumPy and autograd keep the
runtime small. Each arm receives the same initialization, batch order and 512 AdamW updates.
All arms use all-visit supervision. The worker evaluates K=2,4,8 only after training all arms.

Use the `rlt-pilot` extra and a local run directory containing the bounded source snapshots in
`sources/`. Each source JSON includes dataset, config, split, observed repository revision,
response digests and ordered dataset-viewer rows. Text snapshots also carry full-document
hashes and bounded excerpts. The launcher freezes the training and evaluation manifests before
starting the first arm. The reporter checks coverage and initialization and writes item-level
paired scores, per-loop summaries and the sampled resource trace.

Exit code 75 identifies a RAM pause. The launcher writes an atomic optimizer snapshot before
exiting the worker. Re-invoke with the same root to continue from its last committed batch
under the same memory controls. It retains complete evaluation files and redoes an incomplete
file from its first item. A `STOP` file in the run root cancels waiting and stops the pilot.

    python scripts/fetch_rlt_pilot_data.py RUN_ROOT
    python scripts/run_rlt_small_pilot_windows.py RUN_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python -m datagoblin.training_gym.rlt_pilot_report RUN_ROOT EVIDENCE_FOLDER

The text result is bits per UTF-8 byte. The text validation windows come from reserved documents
in the same corpora used for training. HellaSwag uses character-normalized choice likelihood
and ARC uses raw likelihood as the primary score, with both retained. LAMBADA uses greedy
last-word byte exact match. Continuation scoring uses a trailing context and fixed stride.
These are fixed 64-item slices with a byte tokenizer and one training seed. Treat their
results as feedback and transfer diagnostics. The rule pack uses distinct graph lineages,
including length and revision-position holdouts. The second-turn input includes the model's own
first answer, and evaluation labels stay in the separate scorer.

## Graph-conditioned TRM refinement pilot

The graph pilot uses a continuous node state and a separate answer state. One shared 12-unit
nonlinear cell is called first to propose a reasoning update, then to revise the answer. Each
visit propagates the reasoning state across one edge of the supplied adjacency. Only the
feedback coefficient on the reasoning state varies. At the patch boundary the reasoning state
starts again at the query vertex, with the first answer distribution carried into the new turn.
See [the TRM refinement architecture](https://arxiv.org/html/2510.04871v1) for the source of
the two-state motif. This graph cell is a 133-parameter mechanism prototype.

Each of two initialization seeds runs 240 updates per arm on identical 12-episode batches at
K=6. Score the native training/runtime arms at K=4, 6 and 10. At K=6, run all nine
training/runtime combinations. The frozen rule pack preserves earlier test lineages for
retrospective comparison. For confirmatory use, freeze a new pack and keep its evaluation
targets outside the fitting and inference workspace.

For a simplified uniform routing gate with effective coefficient a_k at visit k, the update
operator is `(1-a_k)I + a_k A`, where A routes one edge. At depth K the coefficient of
`A^K` is the product of the K effective coefficients. For a unit gate at K=6 that coefficient
is 1 for direct, 1/720 for RLT, and about 0.00205 for matched damping. Matching the interpolation
coefficient sum matches the expected number of hops in this linear example. The distribution
of hop counts can still differ. The fitted cell has a node-dependent gate, and its exact
answer probabilities also depend on the readout.

Use the `rlt-pilot` extra and the existing frozen rule pack.

    python scripts/run_rlt_trm_windows.py --pack RULE_PACK --out RUN_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python -m datagoblin.training_gym.rlt_trm_report --pack RULE_PACK --run RUN_ROOT --out EVIDENCE_FOLDER

The launcher retains the 256 MiB job cap, 3 GiB physical RAM reserve and single below-normal
CPU worker, and also monitors system commit headroom. Re-invoke after exit 75 to resume a
headroom pause from its saved optimizer state. The scorer checks weight, update, prediction
and pack integrity and emits matched item outcomes, sampled loop traces and run receipts.

## Separate transport from endpoint refinement

The graph pilot supports two placements. Reasoning placement interpolates the one-edge
proposal. Answer placement runs the same reasoning update directly and applies the coefficient
only to the continuous answer proposal. All three feedback modes still use two shared-cell
calls per visit. Use the same frozen pack, model seeds and batch digests across placements.
Direct feedback must produce identical weights and decisions in both. The prior probe adds
an item-seeded, zero-mean perturbation of RMS 0.01 to the continuous answer input at the patch
boundary, then measures visit-wise answer perturbation gain and terminal answer flips.

    python scripts/run_rlt_trm_windows.py --pack RULE_PACK --out ANSWER_RUN --placement answer --probe-prior --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_trm_windows.py --pack RULE_PACK --out REASONING_RUN --placement reasoning --probe-prior --ramcap-package PATH_TO_LEAN_CGRT_HRM400

For a fixed linear endpoint map with eigenvalue `lambda`, RLT carries that eigenmode through
K visits with multiplier `product_{j=1..K}(1 + (lambda - 1)/j)`. Direct feedback has
multiplier `lambda^K`. For real positive `lambda` the RLT multiplier is
`Gamma(K + lambda) / (Gamma(lambda) Gamma(K + 1))`, of order `K^(lambda - 1)`.
Thus the same rule attenuates an expanding mode's exponential growth to polynomial growth,
and slows the decay of a contracting positive mode. These relations require a fixed linear
map or a common scalar linearization at every visit. Time-dependent and non-normal coupled
maps require checking the actual product of Jacobians.

### TRM and HRM placement

Write a boundary update as `z' = F(z,y)`, `y' = (1-eta)y + eta G(z',y)`. With
`A = F_z`, `B = F_y`, `C = G_z'`, and `D = G_y`, its Jacobian is

    [ A       B                           ]
    [ eta C A (1-eta)I + eta (D + C B)    ]

If `B = 0`, changing the answer schedule leaves the transport trajectory unchanged.
With feedback from answer to reasoning, `B` couples the two states and transports prior
errors back into reasoning. For example, the scalar blocks `A=.8, B=.5, C=.6, D=.2`
give a direct Jacobian with eigenvalue about 1.16 despite each uncoupled self-derivative being
below one. An answer-only contraction estimate cannot bound this full system.

For TRM, use this equation at the interval between two answer proposals. In HRM, a candidate
intervention is a slow-state endpoint at a macro boundary. The fast block `F` is then the
composition of its scheduled low-level visits with that slow state held fixed. Give each
arm the same number of fast updates and slow proposals. Set `eta` from the remaining slow
proposals and test changes to the fast budget separately. The matrix motivates probing
both cross-state gains in a frozen host before a trained HRM intervention. See
[TRM](https://arxiv.org/html/2510.04871v1) and [HRM](https://arxiv.org/html/2506.21734v4).

### LDT placement and authority

Apply interpolation to the continuous proposal features, then discretize to a typed candidate
proposal. Use a checked lattice meet for the committed update. A hard candidate set can shrink
under verified information in a fixed epoch. A rule revision creates a new epoch with its own
initial candidate set. For example, a certified singleton from before the patch cannot be
carried through the patch by intersecting it with the new singleton.

The boundary replay feeds each arm's first- and second-turn proposals to the upstream
`certify_and_apply_verified` entry point after the patch. The checker derives the current
endpoint from the public graph and rejects previous-epoch proposals. It verifies a current
proposal before allowing a hard refinement. Counts include acceptance, abstention, epoch
rejection and checker route hops. Checking costs a full traversal in this fixture, so the
replay measures coverage and authority. Its checker cost is the same order as solving the
fixture outright. The typed boundary design comes from the local HybridTRMLDT worktree.

    python scripts/replay_rlt_ldt_boundary.py --pack RULE_PACK --evidence ANSWER_EVIDENCE REASONING_EVIDENCE --upstream HYBRID_TRM_LDT_ROOT --out BOUNDARY_RESULT.json
