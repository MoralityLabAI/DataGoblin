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

## Clocks, route witnesses and native HRM slow cycles

The frozen clock audit varies the learned cell's phase input independently of its step-size
input. With ten actual visits and six training visits, phase can use either `k/10` or
`k/6`, and step size can use either `1/10` or `1/6`. The query hop count, actual visits,
and feedback coefficient `1/(10-k)` are held fixed. The reference phase can exceed one.
The carried-answer intervention compares the inherited answer with a zero answer
initialization on the second turn. The auditor rejects overlapping evaluation lineages.

    python -m datagoblin.training_gym.rlt_clock_audit --pack FRESH_RULE_PACK --source PLACEMENT_RUN_ROOT --out CLOCK_AUDIT

The LDT checker can reuse a verified prefix via `--reuse-route`. Let `x_0,...,x_K` be
the verified old walk and `D` the sources whose outgoing edges actually changed.
Let `j` be the first index below `K` with `x_j` in `D`, or `K` if none exists.
For each `i<j`, the old outgoing edge and new outgoing edge agree at `x_i`. Induction
gives the same walk through `x_j` under both graphs. The new endpoint follows from
exact replay of only `K-j` edges under the patched graph. A no-op or off-path edit can
therefore transfer the endpoint without traversing edges. On a cycle, the earliest
occurrence of a changed source is the one that matters. Cache creation and patch
membership checks carry their own costs. For a witness built solely for reuse, the edge
savings over multiple recertifications are the sum of their reused prefix lengths minus
the cost of the initial walk. The environment owns the old witness and
the final patch, and issues each hard commitment under the current epoch.

There is a scalar limit on what the RLT schedule itself can explain. For a fixed
positive real eigenvalue `lambda`, the preterminal factors
`1 + eta_k (lambda-1)` have the same arithmetic mean in RLT and matched damping.
The factors are positive. By AM-GM the constant preterminal damping maximizes their
product. Both schedules have the same final factor `lambda`. The comparison bounds
gain for that one fixed mode. Network-level correctness also depends on changing
Jacobians, nonlinear routing, and feedback between states.

The published HRM-Text-1B configuration uses two H cycles and three L cycles per H
cycle. At two slow visits RLT and matched damping both use `[1/2, 1]`. A native-budget
comparison can identify the effect of interpolation relative to direct feedback.
A test of the RLT schedule relative to matched damping needs at least three slow
visits. Preserve the same low-level work per slow visit and log the count of both.
The slow-state hook must use the previous carried H state, including the actual
initializer at the first visit. The H block's combined input also contains L state.
[Published HRM-Text-1B configuration](https://huggingface.co/sapientinc/HRM-Text-1B/raw/main/config.json) ·
[HRM-Text implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/hrm_text/modeling_hrm_text.py)

## Learned two-timescale carry pilot

The binary addition pilot uses separate fast and slow cells. The fast cell advances
one column of carry information. The slow cell reads carries and updates the answer
state. The slow state conditions the fast proposal and remains fixed during its two
inner visits. Both cells start from learned four-row tables indexed by the local
operand bits. State updates use continuous probabilities. The model has 24 parameters
and receives neither phase nor normalized step size. Supervision is final-answer bit
cross-entropy. This retains the fast/slow coupling studied in [HRM](https://arxiv.org/html/2506.21734v4)
and gives an exact arithmetic verifier.

Fit direct, slow RLT, slow matched damping, fast RLT and fast matched damping with
the same initialization, examples, and update counts per seed. The selected schedule
span is the full count of visits to its own state. There are four slow visits in
training, with two fast visits per slow visit. Use 2, 4, 6 and 8 slow visits at test.
In the 16-bit challenge, flipping one operand bit at a known anchor initiates or
clears a long carry. Evaluate the second calculation from a fresh state and from the
previous slow state, resetting the fast state both times. Report creation and
cancellation separately. A swap on frozen weights distinguishes a training effect
from the effect of applying interpolation at inference.

For an ideal carry chain without slow-to-fast coupling, when every fast visit is
necessary to generate and move the carry to the far bit, its surviving signal is
the product of the fast feedback coefficients. With 16 fast visits RLT gives
`1/16!` in that control. This is a transport limitation. The slow endpoint state
can be interpolated without directly imposing this product on the carry path.
Learned cross-state dependencies can still couple both trajectories.

    python scripts/run_rlt_hrm_carry_windows.py --out RUN_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python -m datagoblin.training_gym.rlt_hrm_carry_report --run RUN_ROOT --out EVIDENCE.json

The Windows launcher uses one CPU worker, 256 MiB of job commit, and the 3 GiB host
reserve. The evaluator independently verifies binary predictions using integer
addition after inference. A typed incremental checker for a future carry task can
retain certified columns below the first edited bit. It can reuse the old suffix
once the new carry matches the old carry at the boundary of an unchanged operand
suffix. That follows from deterministic per-column transitions, as with the graph
route witness.

## Frozen language-model crossover

The language-model audit reuses the completed three-arm public-benchmark pilot. It
crosses training and runtime feedback at 1, 2, 4, 8, and 16 block visits. One visit
isolates the fitted weights because all feedback rules reduce to the same forward
pass. At two visits the RLT and matched-damping schedules coincide. Text-window
losses use the same FineWeb-Edu and OpenWebText documents across all cells. The
discrete benchmark slices are scored at four and eight visits with native clocks.
The audit also intervenes independently on phase and step-size feature scales at
eight and sixteen visits. The executed depth and feedback coefficients remain fixed.

For a lower-is-better score, let `S(a,b)` denote training rule `a` and runtime rule
`b`. With direct feedback `D` as the reference and `R` the candidate, the training
term is `S(R,D)-S(D,D)`, the runtime term is `S(D,R)-S(D,D)`, and the interaction
is `S(R,R)-S(R,D)-S(D,R)+S(D,D)`. The three terms sum to the joint effect. This is
a decomposition on the frozen matched examples and fitted checkpoints.

An endpoint perturbation at visit `j` enters the next carried state multiplied by
`eta_j`. Later state Jacobians still act on that perturbation. The audit pairs
equal input-RMS perturbations at initialization and at early and late feedback
boundaries, and measures the final raw, normalized, and logit response. Centered
normalization with `z = P h` and `s = sqrt(zᵀz/d + eps)` has derivative
`P/s - zzᵀ/(d s³)`. For centered directions perpendicular to `z`, the gain is
`1/s`. Smaller raw states can therefore exhibit larger normalized responses
even when interpolation attenuates the raw perturbation.

    python -m datagoblin.training_gym.rlt_lm_audit_report --root ORIGINAL_PILOT --audit CROSSOVER_RUN --resources RESOURCES.json --out EVIDENCE.json

The evaluator is an importable function so the operator's hard cap and progress
guard are installed before running it. Per-item native scores are checked against
the original run. The study replays previously viewed slices with one training
seed. A held-out replication requires independent documents and fits.

## Held-out replication across seeds

The replication refits all three schedules with seeds 149 and 191 on the original
training fixture, with 512 AdamW updates at four visits per arm. Fit initialization
and the training batch plan are shared within each seed. The original seed 131
participates in held-out evaluation alongside the new fits. The two additional
fits vary model initialization and batch order while retaining the task mix and
optimization budget.

Freeze 128 fresh 96-byte windows for each text corpus from bounded dataset-viewer
slices starting at row 1000. Reject document hashes already observed in either
corpus and reject candidates whose saved excerpts contain a previously scored
or training window. The manifest binds the fetched slices and resulting windows.
The byte check covers the saved excerpts. The replicas share the new documents,
so report effects by seed and corpus.

The crossover uses native clocks and four depths. At one visit it scores a single
runtime because the schedules coincide. At four, eight and sixteen visits it
crosses all training and runtime rules. Use the per-seed one-visit training
contrast, native RLT versus direct loss, RLT-to-direct swap cost, and matched
damping on the RLT-trained weights. Keep the raw, paired document losses and
training receipt hashes with the conclusions.

For the tied parameters of the repeated block, let `S_k` be the parameter
sensitivity of the carried state. If the schedule is fixed, differentiation gives

    S_(k+1) = [(1-eta_k)I + eta_k F_(x,k)] S_k + eta_k F_(theta,k).

When the candidate derivatives have norm bounds `L` and `B`, the carried-state
bound from parameter injection alone is
`B [product_k (1 + (L-1) eta_k) - 1] / (L-1)`. This follows by telescoping the
geometric weights on the injected derivatives. At `L=1`, it is `B sum_k eta_k`.
Direct then has a linear bound in the visit count and RLT has a harmonic bound.
Any parameter sensitivity in the initial state adds its separately propagated
term. Endpoint and tied-readout gradients also enter the training objective
directly. The first endpoint and its gradient are identical for matched
initialization, batch, and clocks. The initial-gradient probe aggregates the
three later endpoint losses separately and measures their angle with the first.

    python -m datagoblin.training_gym.rlt_lm_replication freeze --root ORIGINAL_PILOT --sources FRESH_SNAPSHOTS --out HOLDOUT
    python -m datagoblin.training_gym.rlt_lm_replication score --run REPLICATION_EVAL --holdout HOLDOUT --out EVIDENCE.json

## Released Sudoku HRM and an LDT stopping boundary

This checkpoint test uses the released 27M-parameter Sudoku Extreme HRM. It
retains the checkpoint's 16 supervised segments, two slow visits per segment,
two fast visits per slow visit, and its separate recurrent states. Feedback can
act on the slow state across the two visits within each segment or across all
32 visits. The two-visit schedule repeats `[1/2, 1]`. Global RLT uses the
remaining slow-visit count across the full rollout. Global matched damping
controls for global RLT's coefficient sum and terminal full update. It has a
coefficient sum of about 4.06 across 32 visits. The segment-reset comparison
uses a sum of 24, and direct uses 32. Comparison between global and segment
scopes therefore changes the schedule and the total interpolation dose.
[Released checkpoint and config](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme/tree/fbb4eb8)

The squared norm of an interpolated state has an exact decomposition. For
`h' = (1-eta)h + eta h_hat`, it is the weighted average of the two squared
norms minus `eta (1-eta) ||h_hat-h||²`. Even when the post-normalized endpoint
has unit RMS, interpolation can reduce the magnitude of the state injected
into the next fast stack. Tracking both the prediction and carried-state RMS
helps detect this change.

The CPU-only Torch preflight reproduced the upstream slow state, fast state,
and logits at every segment for a reference puzzle. The Torch process crossed
the 256 MiB private-commit target, so the benchmark uses a bounded NumPy
backend with bfloat16 rounding at operation boundaries. Its weights are decoded
from the pinned release ZIP with a restricted layout reader. A cross-backend
gate checks segment predictions and final boards against the two saved CPU
reference rollouts before the benchmark is scored. The official model uses GPU
FlashAttention, so this check pertains to the CPU reference implementation.
[Upstream HRM source](https://github.com/sapientinc/HRM)

The fixed holdout is a deterministic 24-puzzle slice of the public Sudoku
Extreme test split. All arms execute the full planned budget during inference.
Record both exact match and Sudoku validity at each supervised segment. The
strict upstream LDT boundary independently certifies a proposed grid by checking
its givens, rows, columns and boxes. A verified stopping replay accepts the first
valid grid. Compare retained scores and estimated stack calls with full-budget
inference, keeping the checker calls explicit. This check evaluates a proposed
solution without a search for candidates. [Official test split](https://huggingface.co/datasets/sapientinc/sudoku-extreme)

For a fixed forecast sequence, the verified exit time is the first segment
whose grid passes the checker, or the budget if every segment fails. The gate
accepts a valid solution whenever any prefix contains one. The model still
generates candidates, and the checker carries responsibility for acceptance.
Earlier acceptance is useful even for trajectories that later leave a valid
grid. Exact equality to a dataset label remains a separate metric when a
valid solution could differ from the stored label.

    python scripts/run_rlt_hrm_sudoku_windows.py --run PINNED_RUN --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python -m datagoblin.training_gym.rlt_hrm_sudoku --run PINNED_RUN --out EVIDENCE.json --upstream-ldt HYBRID_TRM_LDT_ROOT

The run directory contains the public checkpoint, release config and provenance,
the frozen holdout, and the archived CPU-reference predictions. Keep generated
weights and private evaluation files outside the repository.

## Attention TRM on the same Sudoku holdout

The checkpoint sweep applies five inference rules to one frozen, public
community attention TRM checkpoint. It uses the upstream Sudoku attention
recipe with 16 supervised segments, three answer updates per segment, and six
reasoning updates before each answer update. A shared two-layer stack proposes
both kinds of update. All arms execute 21 stack calls per segment, and both
states carry across supervised segments. [Upstream TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/c01103738605ba39d1430519b1ee0c62f4c707f8)
· [Checkpoint](https://huggingface.co/yagizdevre/trm-sudoku-extreme-att/tree/56e0d4ac6d91671c60cc1314bfef6c3f620693f2)

Answer-state RLT resets its clock across each group of three answer updates.
Its coefficients are `[1/3, 1/2, 1]`; the matching control uses
`[5/12, 5/12, 1]`. Reasoning-state RLT resets its clock for each group of six
reasoning updates. Its control matches the coefficient sum and the final full
update in that group. In both placement comparisons the other state uses direct
feedback. The schedules have the same stack calls. Since both states enter
the shared stack, changing either carried state can affect both trajectories.

For a scalar affine endpoint with fixed slope `lambda`, one answer update
has gain `1 + eta (lambda - 1)`. Across three answer updates, the RLT gain
minus the matched-damping gain is exactly `-lambda (lambda - 1)² / 144`.
Matching the coefficient sums cancels the first-order terms. This calculation
isolates the temporal schedule with a fixed endpoint derivative. In the
coupled loop, collect six direct fast visits as `z' = A z + B y`, and write
the slow endpoint as `C z' + D y`. The complete cycle matrix has the form
`M(eta) = M0 + eta M1`, where `M0 = [[A, B], [0, I]]` and
`M1 = [[0, 0], [CA, CB + D - I]]`. Swapping two coefficients changes the
product by `M(b)M(a) - M(a)M(b) = (b-a)(M1 M0 - M0 M1)`. This term depends
on coefficient order and the cross-state derivatives.

The Windows worker uses one CPU-only Torch thread under a hard 256 MiB
Job Object and the same host-memory floor as the HRM test. It checks every
direct segment board against two frozen reference traces before scoring.
Use the separate reference script included with the run receipts to generate
these traces. The LDT gate independently validates candidate grids at segment
boundaries. The checkpoint sees givens alone, while exact-match scoring happens
after inference. The archived replay counts checks and projected stack calls.

The HRM and TRM sweeps share puzzle IDs and labels. Compare per-segment scores
with the model-specific layer and token costs. The compute audit estimates
stack projection and dense attention multiply-accumulates and compares the
models at a shared ceiling. Observed differences also reflect the two
checkpoints' distinct training and numerical backends.

    python scripts/run_rlt_trm_sudoku_windows.py --run PINNED_RUN --ramcap-package PATH_TO_LEAN_CGRT_HRM400 --torch-cpu CPU_TORCH_ROOT
    python -m datagoblin.training_gym.rlt_trm_sudoku --run PINNED_RUN --out EVIDENCE.json --upstream-ldt HYBRID_TRM_LDT_ROOT

## Live LDT stopping on rule changes

The graph-conditioned TRM exposes its top answer after each completed visit.
The strict LDT membrane commits it only after independent exact verification
in the current evidence epoch. A commitment immediately ends the loop and
the first turn passes its accepted prefix's answer distribution to the
second turn. The RLT schedule and depth inputs retain the original budget
throughout that prefix. If the budget expires without a commit, the verified
policy abstains. The fixed baseline runs both turns to completion and emits
its terminal guesses.

The verifier constructs the original route once, then checks a patch by
resuming from the first affected departure on that route. The second-turn
gate verifies in a fresh epoch. A separate policy can submit the earlier
certified answer at visit zero when the patch leaves the graph identical.
It commits only after checking the new epoch. The verifier's route is
isolated from the neural proposal and its answer distribution.

The comparison freezes existing answer-placement weights and evaluates
direct, RLT and matched damping at six and ten visits per turn. The new pack
contains both actual edits and no-ops on the same lineages. The long cohort
balances seven-, eight- and nine-hop queries. Exclude lineages from all
supplied previous packs, including the checkpoint training pack. Report
per-item exact answers, abstentions, live neural visits, membrane checks and
independent route hops. The exact route checker can solve this fixture by
itself, so its work belongs in any claim about the hybrid's cost.

    python scripts/run_rlt_trm_ldt_windows.py --out RUN --checkpoints ANSWER_RUN --previous-pack TRAINING_PACK --previous-pack PRIOR_HOLDOUT --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400

## Live LDT stopping on fresh Lean core goals

Reuse the three matched RLT-02 Loop-XS checkpoints at 3,400 updates.
Score a shared eight-tactic portfolio using the byte model's mean
log probability, including EOS, at each encoder visit. Propose the
highest-ranked remaining tactic and halt as soon as the strict LDT
boundary commits a proof. The four- and eight-visit trajectories use
their own planned horizons. The native baseline runs the same
portfolio in its preregistered fixed order under the same tactic limit.
At the eight-attempt budget every arm can reach the entire portfolio.

The verifier owns the Lean root and takes each tactic from that root.
It asks Lean for an explicit kernel check only when the result has no
remaining goals, then checks the exact theorem's axiom footprint.
The timing and counts distinguish encoder visits, tactic requests,
and explicit kernel-check requests. Shared goal initialization and
the separate standalone replay are recorded outside arm latency.
Replay the unique successful theorem–tactic pairs through the Lean
frontend to cross-check the live gate.

The fresh fixture contains generated Lean 4.19 core propositions.
Weights stay frozen. The suite probes tactic ordering and verified exits
on this grammar. The CPU Torch phase checks score and ranking parity
with the NumPy inference path before the Lean evaluation. Both phases
and the standalone compiler replay use the hard 256 MiB Windows cap.

    python scripts/run_rlt_lean_ldt_windows.py prepare --out RUN --checkpoints MATCHED_LOOP_ROOT --torch-cpu CPU_TORCH_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_lean_ldt_windows.py evaluate --out RUN --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_lean_ldt_windows.py certify --out RUN --ramcap-package PATH_TO_LEAN_CGRT_HRM400

## Grounded fallback on harder Lean goals

Compose two or more core lemma applications in freshly instantiated list
and arithmetic goals. The native portfolio runs first on every frozen
goal. Only goals left open enter the fallback comparison. Derive candidate
names from a source-pinned core lemma catalog, scoring the names against
operations in the rendered target. Build a single pool of explicit
\`simp\`, \`rw\`, \`exact\`, and \`apply\` proposals. The lexical heuristic and the
three Loop-XS schedules receive that same pool and six attempts each.
A wider lexical pass measures the pool's reachable coverage.

Each ranked proposal restarts from the original root and the live LDT
gate stops immediately on a kernel-checked proof. Count retrieval
queries, lemma comparisons, encoder visits, tactic RPCs and explicit
kernel RPCs. Attribute the shared native prefix and retrieval once to
each end-to-end policy when reporting its work. Compile the distinct
winning proof scripts and the fixture's withheld construction proofs
separately. The latter establish that the frozen targets are provable.

Use a new run directory with the same \`prepare\`, \`evaluate\` and \`certify\`
commands above, adding \`--grounded\` to select the fallback experiment.

## Train the grounded ranker on checked outcomes

Freeze three Lean cohorts before collection. Use 108 goals for labels,
36 new instances of the training templates for one holdout, and 24
instances of four withheld compositions for transfer. Check every
grounded candidate independently at the training root. Record kernel
certificates, failed attempts and inconclusive results. Keep the
construction proofs out of candidate generation and optimization.

The compact ranker scores the same pool as lexical ordering. Its features
come from visible goal operations, tactic forms, and retrieved lemma
operations and positions. Each visit updates a candidate state with a
shared cell that reads the candidate and the mean states of its pool.
Direct, RLT, and matched damping share architecture, initialization
and batch order within each seed. The loss averages the negative
log-probability of the verified positive set across the planned visits.
Inconclusive labels are masked. Training uses pools with a known positive
and negative. Pools without a verified candidate remain useful receipts
for measuring retrieval coverage.

At inference, run the native portfolio once and compare fallback policies
on the same stalled goals. Rerank untried candidates once per visit. The
strict Lean gate commits the first current proof that passes the kernel.
Log retrieval, scoring, and verification separately. A wider untrained
pool pass estimates candidate reachability. Standalone compilation audits
the successful scripts and withheld construction proofs.

    python scripts/run_rlt_lean_ldt_windows.py collect --trained --out RUN --previous-fixture PRIOR_GROUNDED_FIXTURE --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_lean_ldt_windows.py fit --trained --out RUN --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_lean_ldt_windows.py evaluate --trained --out RUN --upstream-ldt HYBRID_TRM_LDT_ROOT --ramcap-package PATH_TO_LEAN_CGRT_HRM400
    python scripts/run_rlt_lean_ldt_windows.py certify --trained --out RUN --ramcap-package PATH_TO_LEAN_CGRT_HRM400

Use the optional pilot dependencies for fitting and scoring. Each Windows
phase uses one CPU worker, a 256 MiB process-tree cap and the same host
reserve. Completed labels and traces are atomic, and optimizer receipts
allow a paused training arm to resume at its saved update.
