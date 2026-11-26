# MOLT-GOBLIN-01: experiment-array gym and curriculum

Status: development proposal with an executable offline fixture. Source inspiration:
[Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923). The implementation here is an independent
experiment-planning fixture; method reproduction would require a separate source-to-code audit.
Companion work uses branch `research/molt-goblin-01-20261126` in LoopedTransformers, HybridTRMLDT,
BPD and Compactification-Theory. Existing [training-gym](training_gym.md), RLT and LDT-array contracts continue
to govern their own evaluations.

## Purpose and current boundary

Train DataGoblin to propose a useful experiment, identify the controls it needs, and select an affordable
array of experiments from verified receipts. A learned producer emits candidates. The LDT sorter recommends
an admissible array or abstains. An operator-controlled executor runs approved recipes, while an independent
steward scores the resulting evidence. The producer's role grants proposal authority only.

The current module, `datagoblin.training_gym.molt_goblin`, generates deterministic JSONL problems and exact
answers. It exports train-only chat SFT and scores external predictions. Checkpoint adapters, interactive
episodes, resource metering and native Verifiers v2 integration remain proposed work. The current score
always carries `promotion_authorized: false`.

| Layer | Present behavior | Next concrete addition |
|---|---|---|
| Problem generator | Four finite synthetic families with reproducible root IDs | Paired repairs, delayed evidence and source-changing episodes |
| Solver and scorer | Exact family answers; digest binding; missing predictions count as failures | Schema diagnostics, unsafe-selection counts and episode-level outcomes |
| Model interface | JSONL predictions supplied externally | Revision-pinned adapter and bounded observation/action interface |
| Training export | Train split receives answer-only chat SFT | Verified train-split action/receipt traces and curriculum selection |
| Research evaluation | Fixture correctness | Frozen comparisons with measured costs and independent root uncertainty |

## Runnable fixture contract

Each public task contains `schema`, `task_id`, `root_id`, `split`, `family`, `problem` and `instruction`.
The generator uses four to six candidates or positions for train, calibration and ordinary test; its current
OOD setting uses seven to nine. This OOD axis measures size transfer. Semantic family holdouts require an
additional generator and a prospective release.

| Family | Decision to learn | Exact answer |
|---|---|---|
| `parallel_array` | Choose qualified, prefix-valid concurrent candidates under summed RAM, CPU-slot and visit caps | Selected IDs and maximum utility; ties use the lexicographically smallest sorted ID tuple |
| `pareto_array` | Compare uncertainty intervals on maximizing axes; cost is represented with its sign reversed | `INELIGIBLE`, `DOMINATED`, `UNRESOLVED` or `SELECTED` for each candidate |
| `prefix_dag` | Detect edges from a position to itself or an earlier position; otherwise compute dependency depth | Invalid edges, or maximum number of vertices on a directed path |
| `matched_control` | Find shams with the same root, parameter count, tokens, visits, verifier calls and dose | Matching sham IDs plus structural `READY` or `REDESIGN` |

In `pareto_array`, conservative dominance compares the contender's lower bound with the candidate's upper
bound on every axis. Possible dominance compares the contender's upper bound with the candidate's lower
bound. Both require a strict improvement on at least one axis. `UNRESOLVED` preserves the need for further
evidence. `SELECTED` is the fixture's interval classification, without a launch decision attached.

The `parallel_array` resource values are synthetic problem inputs. Its 8,192-unit RAM cap and eight CPU slots
have no authority over the host's runtime limits. Utility is supplied by the fixture; forecasting useful
scientific outcomes is a later task. The control family's `READY` means that an exact structural match exists.
Statistical resolution and permission to run have their own gates.

For example, a five-position prefix problem with edges `[[0, 2], [1, 2], [2, 4]]` has answer:

```json
{"invalid_edges": [], "stabilization_bound": 3}
```

Adding `[4, 2]` produces `{"invalid_edges": [[4, 2]], "stabilization_bound": null}`. This creates a useful
paired exercise: the size and most of the graph stay fixed while one edge changes the valid conclusion.

Generate fresh directories and score predictions using the existing CLI:

```sh
python -m datagoblin.training_gym.molt_goblin generate --split train --seed 71 --per-family 128 --public runs/molt/train --steward runs/molt/train-labels
python -m datagoblin.training_gym.molt_goblin generate --split calibration --seed 91 --per-family 32 --public runs/molt/calibration --steward runs/molt/calibration-labels
python -m datagoblin.training_gym.molt_goblin score --public runs/molt/calibration --steward runs/molt/calibration-labels --predictions runs/molt/predictions.jsonl
```

Predictions contain one `task_id` and its `answer` per JSONL row. Unknown or duplicate prediction IDs raise
an error. The scorer checks the public release digest and row count, matches the label roster, then recomputes
each exact answer from its bound input. It reports correct/total counts per family and the number missing.
The generator refuses an existing release directory. Richer diagnostics in the table above remain unimplemented.

## Proposed episode interface

Keep fixture v1 intact and add a separately versioned episode adapter. Its first implementation should replay
saved outcomes offline through the same transition function that a future live executor would use.

| Record | Required fields and owner |
|---|---|
| `TaskEnvelope` | Steward supplies root/lineage ID, split, task version, `frame_id`, `problem_version`, `parent_frame_id`, `premise_digest`, public observation, permitted actions, budget and verifier version |
| `CandidateRecipe` | Producer supplies hypothesis, falsifier, exact source/checkpoint/recipe digests, proposed control, predicted resource interval and requested evidence |
| `Action` | Adapter emits a typed `propose`, `select`, `request_evidence`, `repair` or `abstain`, bound to the current frame, within-frame `epoch_id` and public-state digest |
| `ExecutionReceipt` | Executor records run ID, frame identity, recipe digest, status, measured resource use, output digest and cleanup outcome |
| `VerificationReceipt` | Independent verifier binds the frame, input and execution receipts to a verdict, evidence class and permitted public-state update |
| `EpisodeSummary` | Steward records final decision, all charged actions, budget termination reason and whole-episode score |

The names above describe a proposed interface, with no new schema implied in the current module. Canonical
content bindings are `input_digest`, `public_state_digest`, `checkpoint_digest` and `verifier_digest`.
Semantic labels such as `checker_version` remain separate from these content digests. `frame_id` and
`problem_version` identify the premises; `epoch_id` identifies a frozen computation snapshot within that frame. The adapter will
use a separately versioned schema, while existing fixture and core serialization formats retain their keys.
The adapter must reject unknown fields at the authority boundary, malformed IDs and graph indices, stale state digests,
duplicate task IDs, and actions outside the envelope. A candidate's claim to be verified stays a claim until
the environment creates a verification receipt.

An episode begins with an immutable public envelope for one problem frame and a declared budget. Each
action consumes budget before the environment returns a receipt. Within a frame, accepted refinements
advance the public LDT state monotonically and increment its epoch. Rejected proposals and cap aborts become
immutable events. A terminal decision or exhausted budget closes the episode, after which the steward scores
the complete trace. Evaluation reports keep authoritative labels outside the training process.

A source change that retracts evidence or changes the candidate domain requires an environment-owned new
problem frame. The environment rebuilds its initial candidate state from premises valid in that frame,
records the parent frame and closes the old frame. Every envelope and receipt remains immutable; revisions
are appended. Cache invalidation accompanies the transition, while the new frame explicitly permits
reconsidering candidates eliminated under the earlier premises. Both frames retain the same root lineage
and split, and all spent work remains charged to the episode budget.

An illustrative delayed-evidence episode would proceed as follows:

1. The observation contains two candidate recipes with similar predicted utility. One relies on source version A.
2. A requested evidence action returns an independently verified receipt for version A and charges one verifier call.
3. Several intervening observations arrive. A later environment receipt records a relevant source change to
   version B, opens a new frame and reconstructs its candidate state from the surviving valid premises.
4. The final array decision must use evidence valid in the new frame or obtain a fresh receipt within the
   remaining budget. Candidates eliminated using the retracted version-A evidence can be reconsidered there.

Pair this episode with one in which an unrelated source changes. Both have similar surface edits; only the
first requires rebuilding the affected candidate state. Score correct final selection together with receipt
validity, the correct frame transition and budget use.
For a memory experiment, freeze what history can be reread and charge every retrieval. Showing the complete
trace at the final step would answer a different question about reading a long context.

## Curriculum progression

| Stage | New task pressure | Environment-owned check | Evidence needed to advance |
|---|---|---|---|
| 0: exact arrays | Qualifications, interval ambiguity and tight budgets | Current solver answers | Reproducible releases and correct scoring; exact-policy ceiling retained |
| 1: repair and discrimination | Find a minimum valid edit; choose between explanations that predict different receipts | Enumerate a bounded edit set and replay paired counterfactual worlds | Repairs restore the required invariant, with minimality checked under a frozen edit metric |
| 2: delayed state | Use old receipts after distractors, source changes and restricted rereads | Versioned dependency graph and exact event replay | Whole-episode validity and recovery improve under a declared memory/retrieval budget |
| 3: research transfer | Select a bounded LBF search, TRM/LDT recipe or BPD intervention from public candidate records | Domain verifier plus measured execution receipt | Fresh task lineages, ordinary and OOD results, complete cost reporting and prospective preflight |

Proposed repair tasks include removing a forbidden graph edge, swapping an over-budget candidate and
replacing a sham with one that matches the physical dose. Each task needs a frozen edit vocabulary and
tie rule. The generator should pair successful repairs with plausible edits that leave the relevant defect
in place, so a model must identify which constraint caused failure.

For delayed state, vary dependency depth, the distance to a required receipt, distractor count and the location
of an invalidating edit. Use multiple independently generated roots at each setting. All transformations of
one underlying world inherit its lineage and split. Memory interventions include a fresh cache, an empty
cache, a shuffled cache and a stale-version cache. Preserve the same admitted evidence and charge differences
in access cost.

Research transfer can then reuse the typed envelope for Jinn/Beast membrane exercises, externalized persona
state, BPD control selection and LBF proof-search allocation. Each domain must provide its own executable
verifier. A generic utility number in a planning fixture supplies no evidence that a chosen architecture learns
better or that a proof-search policy closes more goals.

## Comparison plan and resource accounting

The checkpoint roster is baby Goblin, Dory 4B, Nanbeige 3B, Qwen 4B, HRM Textual 1B and Inky Mini, followed
by future checkpoints with pinned local revisions. Missing adapters or local weights receive `unsupported`
in the run manifest. Keep frozen-base, random-policy, exact-policy, typed-LDT and learned-array controls.

For checkpoint comparisons, deliver the same task information and freeze the output schema, tool access,
episode limits and truncation rule. Record each model's tokenization and emitted tokens. Native recurrent
passes need a separate counter. A proposed initial envelope is 2,048 input tokens, 512 generation tokens,
eight turns and four experiment requests; tokenizer differences require an explicit common-information policy
before these values become a frozen comparison contract.

Architecture comparisons require two views. A fixed-data panel holds examples, processed tokens and the
update schedule constant while charging extra recurrence. A fixed-compute panel lets the simpler control
spend the saved budget under a preregistered schedule. Report both panels with their actual charges. The
same parameter count, token count and update count alone cannot make one pass and multiple passes consume
equal compute.

Every run receipt should record total and trainable parameters, non-padding training tokens, optimizer
updates, forward and backward block-token visits, generated tokens, verifier calls, CPU time, wall time,
peak resident RAM, peak accelerator allocation where applicable, and all cache construction/refresh work.
Unsupported meters should remain explicitly missing. Budget planning can use block-token visits; measured
time is still needed because attention length and cache access change cost.

The existing tiny/mock training contract sets 2,048 MiB RAM, 50% CPU, 50 MB/s I/O, 900 seconds and zero swap.
Real checkpoints require an operator-provided hard-cap wrapper and the existing training-gym approval path.
A 2/4/8/16 GiB memory study is a proposed later cap sweep. Each approved cap covers the whole worker,
including weights, optimizer, active tensors and cache. Hold CPU allowance and declared run budget fixed,
and count cache rebuilds. Larger caches earn a useful result only through measured learning or inference
efficiency on the declared task.

## Split custody and curriculum feedback

The steward chooses fresh private seeds for `ordinary_test` and `ood_test`. Keep steward files outside
builder access with an external access policy; separate paths inside one process provide plumbing separation.
Public algorithmic answers remain derivable from the problem by design. Train SFT preserves task and root
IDs, and exports only the train split. Generated releases and private seed material stay out of git.

Hash namespaces identify provenance. Audit semantic duplicates, inherited templates and graph isomorphisms
before claiming independent task transfer. A proposed lineage manifest should include parent world IDs,
generator version, transformation family and split inheritance. Assign the world to a split before deriving
repairs, paraphrases or counterfactual variants.

Curriculum feedback uses verified train-split failures and receipts. Retain invalid, adverse and aborted
attempts with their status, then add fresh train roots that exercise the failing constraint. Calibration may
select a checkpoint or schedule under its registered role. Ordinary and OOD releases remain evidence-only.
Export public actions, executable checks and observed receipts; private hidden reasoning traces are outside
the training record.

For Jinn/Beast specialist data, retain the authority boundary in every example: the proposer supplies a
candidate, the environment verifies it, and the sorter ranks the candidates that satisfy the declared
constraints. Add train-only counterexamples in which a persuasive candidate fails a concrete verifier rule.

## Repository handoffs and milestones

| Owner | Artifact it supplies | Receiving gate |
|---|---|---|
| DataGoblin | Versioned public tasks, train exports and steward-scored receipts | Adapter parity with the standalone scorer |
| HybridTRMLDT | Public-state/latent-memory contract, problem frames and epoch validity checks | Verified monotone updates within a frame; new valid initial state after premise changes; correct cache invalidation |
| LoopedTransformers | Matched architecture runs and complete resource receipts | Frozen budget panel, all attempted arms and reproducible checkpoint identity |
| BPD | Component/donor definitions and physically matched interventions | Reconstruction parity, dose calibration and independent development shams |
| Compactification-Theory | Assumptions and exact dependency claims | Fixture instantiation satisfies the stated graph and map conditions |
| EvalMath | Resolution preflight and paired/clustered inference | Independent roots, frozen contrasts and an untouched confirmation release |

The next adapter milestone passes when exact-policy predictions agree with the current scorer and malformed
or stale actions receive deterministic rejection. The episode milestone requires a replayable transition
trace whose final decision and charged budget can be checked independently. A later model milestone needs
a paired development comparison with failures retained and resource meters complete. Research confirmation
requires its own frozen release and the EvalMath preflight outcome.

Each research handoff must include the independent unit, `n_units`, `n_cases`, primary contrast, physical
dose, scientific effect floor, matched-sham floor and its source/confidence, planned and actual comparisons,
estimated resolution/power, readout exposure, causal controls, verdict and exact evidence provenance.
Generated rows, recurrence steps and alternative doses from the same world count as observations within
that world. Report model-seed and task-root variation according to the claim being tested.

Retain family accuracy, exact selected sets, unsafe selections, abstention, resource use and whole-episode
success as separate outcomes. Present the resulting Pareto frontier. Solver saturation closes the fixture's
usefulness for an architecture contrast; the next discriminator needs fresh roots and a task that makes the
proposed mechanism matter. Promotion stays disabled until the broader training-gym requirements are met.
