# Paired Looped Transformer curation study

This experiment uses the actual upstream LoopedDecoderLM from
LoopedTransformers at 950d4d594d87a1f74c6139a928a9b6cb4ab70617.
Both arms use the same structured slot tokenizer, one physical self-attention
block repeated three times, and the same shared pointer head. Per seed the
initial weight hash is identical. Both arms sample the same indices within
matched-length presentations and make the same number of AdamW updates.

Each TRAIN root has one initial selection and a bounded set of distinct
membership-edit rollouts. Candidates include an exact canonical repair,
alternative endpoint selections and alternative edit orders. Alternatives
admit realistic near-misses under a shared rollout length and STOP label
format. This produces earlier stopping relative to the optimum when an
alternative ends at a suboptimal selection. Each curator sees the same
rollout facts and emits one offer per passing candidate in the same schema.
Good MeTTa checks independent public constraint features for revision,
lineage, dependencies, conflict, size, cost and eligibility, and ranks
distinct goal coverage. Sloppy MeTTa applies weaker local constraints and
ranks raw goal mentions. Ranking, tie-breaking, and the one-rollout-per-root
quota are identical outside the rule files. A quota shortfall excludes the
root from both training arms.

Independent exact replay audits both chosen datasets before training.
The audit uses verifier labels, which the tokenizer and optimizer never
read. The supervised labels are the original actions of the selected
rollout. Reported supervised tokens count one target action per state;
each arm uses the same fixed-width input token encoding.

Both trained policies use the same typed public membrane and revisit
filter at evaluation. The independent Verifiers reward receives only
accepted actions and checks safe completion and correct STOP. The report
counts policy verifier calls, vetoes and recurrent passes separately
from grading checks. Applied unsafe additions are constrained by the
shared membrane, while vetoes expose rejected proposals.

## Reproduce

Use Linux or WSL, CPU-only PyTorch, and the pinned upstream Git checkout.
Run from the DataGoblin checkout with both repositories on PYTHONPATH.
Keep the checkpoint directory in scratch space outside the Git worktree.

    python -m pip install -r metta_research_bench/requirements.txt
    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    export PYTHONPATH="$PWD:/path/to/pinned/LoopedTransformers"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
    vf-validate metta-research-bench --taskset.split looped_dev --taskset.roots-per-family 12 --runtime.type subprocess -n 60
    python -m metta_research_bench.looped pilot --out /tmp/metta-looped-pilot --checkpoints /tmp/metta-looped-pilot-weights
    python -m metta_research_bench.looped train --out runs/metta-looped --checkpoints /tmp/metta-looped-weights
    python -m metta_research_bench.looped frozen --out runs/metta-looped --checkpoints /tmp/metta-looped-weights

Protocol settings are in protocol.json. Training and development must
finish before the frozen subcommand can load any frozen root. It checks
the committed source, rule hashes, protocol hash and checkpoint hashes
against the locked development report. The primary endpoint is measured
on fresh in-family roots. Conflict transfer remains separate. A
root-clustered bootstrap pairs arms and retains all three seeds for every
sampled root. Evaluation allocates a shared ten-decision cap, with a fixed
three recurrent passes per decision and a shared four-proposal cap.

## Recorded result

The pool retained 10,662 distinct candidate rollouts from 240 TRAIN
roots. Each curator selected one rollout per root with zero shortfalls.
Selection differed on 133 roots. The independent audit measured 236/240
safe completions for good MeTTa and 107/240 for sloppy MeTTa. The sloppy
selection logged 119 unsafe additions involving duplicate lineage,
missing prerequisites, stale revisions, or duplicate revision keys.
Four good selections still stopped early because the fixed-length
alternatives did not safely reach the exact dependency optimum.

All six fits used the same 44,417-parameter architecture and 600 AdamW
updates, with 28,800 supervised action tokens per fit. Each seed pair
began with identical weights. A rerun after the lock hash correction
reproduced all six checkpoint hashes, the candidate pool hash, and the
curation and protocol hashes before either frozen split was opened.
Gold and noop validation passed on all 60 development tasks.

| Split | Good MeTTa | Sloppy MeTTa | Paired gain | Root-clustered 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Development | 63/180 | 44/180 | +10.56 pp | See locked development report |
| Fresh in-family primary | 190/600 | 147/600 | +7.17 pp | +2.50 to +12.00 pp |
| Fresh conflict transfer | 23/96 | 10/96 | +13.54 pp | +2.08 to +25.00 pp |

Each arm evaluated the same 200 primary roots and 32 conflict roots
under all three seeds. Primary seed-paired gains were +11, +6, and +4
percentage points. The primary result met the preregistered five-point
threshold with an interval above zero and zero applied unsafe additions
in either arm. The shared typed membrane enforces the last condition.
On the primary split, good curation reduced premature STOPs from 407
to 388, vetoed proposals from 583 to 142, policy verifier calls per
success from 19.37 to 12.04, and recurrent passes per success from
57.94 to 43.07. Exact first edits fell from 201 to 194.

The training and scoring evidence is tied to DataGoblin commit 74e176f.
The run reports include the locked development manifest, selected
rollout IDs, independent audit, per-seed outcomes, and 1,752 Prime
Verifiers traces across all three splits.
The recorded frozen report has SHA-256
a1da373dd35253991671b9fad665200ab05207d1be4821f37db826463c0bf5cc.
The CI workflow runs the development pilot in runner scratch directories.
Recorded checkpoints and generated traces remain in local run directories.

## Action-set supervision transfer

`python -m metta_research_bench.looped.action_ties --archive /path/to/outputs --out /path/to/fresh-repair-ties-run`
compares three same-size Looped Transformers trained on identical
public curation states. One arm labels the canonical safe next edit,
one uses a source-hash-selected safe edit, and one distributes target
mass over all shortest safe edits. Every edit in a target set must pass
the environment's typed proposal check and reduce the shortest remaining
gated path by one. At the verified terminal state the only target is STOP.
The native `action_ties.metta` rule reads environment receipts and
checks the exported target set. Reverse breadth-first search uses actual
typed verifier decisions as edges. This includes repairs that must
temporarily remove a record retained in the final optimum because an
add cannot leave an unresolved dependency.

Training uses 240 roots and two alternative shortest trajectories per
root. Both label views come from the shared states in those trajectories.
Seed-paired weights, sampled batches, action masks, and 240 AdamW
updates are matched. The primary contrast is action-set versus canonical
labels on 400 fresh roots across the five in-family generators. The
hashed singleton control distinguishes the effect of crediting multiple
actions from the effect of using a different fixed tie-break. A separate
80-root conflict cohort remains outside training and selection. Both
cohorts use the original typed inference gate and independent Prime
Verifiers reward. The runner locks source hashes against 19 archived
files and writes weights' digests, training frames, and frozen traces to
the output folder for local audit.

| Label target | In-family completions | Policy verifier calls | Premature stops | Conflict completions |
| --- | ---: | ---: | ---: | ---: |
| Canonical safe edit | 441 / 1,200 | 5,079 | 683 | 71 / 240 |
| Hashed safe edit | 474 / 1,200 | 4,458 | 681 | 73 / 240 |
| All shortest safe edits | 507 / 1,200 | 4,483 | 651 | 76 / 240 |

In-family action-set supervision gained 5.50 percentage points over the
canonical target (descriptive paired 95% root interval +2.75 to +8.33)
and 2.75 points over the hashed target (+0.25 to +5.08). The primary
contrast saved 596 policy verifier calls and 1,305 recurrent visits.
On trials initially requiring at least two gated edits, action-set
supervision completed 449/1,128 against 388/1,128 for the canonical
target. The conflict contrast versus canonical was +2.08 points
(−3.75 to +7.92). All evaluated policies applied zero unsafe additions.

The corrected run used DataGoblin `2637923` and LoopedTransformers
`950d4d5`. Each of nine fits used 44,417 parameters, 1,560 shared states,
and 11,520 sampled action targets. The native rule replayed all 1,560
training action sets, 620 of which had multiple equally short actions.
Two training states needed a detour even though the final membership
was closer by Hamming distance. A full directed-graph audit independently
verified those shortest sets and replayed all 4,320 Prime Verifiers
trials. The run took 55.7 CPU wall seconds and peaked at 468.07 MiB.

## Frozen STOP policies

`python -m metta_research_bench.looped.stop_policy --prior /path/to/action-ties-verified --out /path/to/stop-policy-run`
recreates the three action-set checkpoints from archived states and verifies
their digests before scoring. A development-only selection chooses one additive
STOP logit offset. The selection maximizes verified completion with no increase
in unsafe additions and at most 15% more policy verifier calls. A fresh frozen
comparison crosses this offset with a bounded native STOP witness. Prior
conflict, training, and frozen sources are excluded from both new cohorts.

The witness examines unvisited one-edit neighbors with strictly better public
rank. It can defer STOP only when a typed check accepts a feasible successor
and MeTTa returns the corresponding rank certificate. It shares the original
four-proposal cap and uses at most two verifier calls per episode. The experiment
logs rank probes, typed verifier receipts, native queries, and recurrent visits.
The original Prime Verifiers reward still requires a safe exact repair and an
explicit STOP. The primary contrast is the witness alone versus the unmodified
policy. Completion on the separate fresh conflict cohort remains exploratory.

| STOP policy | In-family completions | Premature STOP | Verifier calls | Conflict completions |
| --- | ---: | ---: | ---: | ---: |
| Unmodified | 485 / 1,200 | 658 | 4,720 | 92 / 240 |
| Calibrated offset | 494 / 1,200 | 640 | 4,877 | 93 / 240 |
| One-edit witness | 485 / 1,200 | 658 | 6,968 | 93 / 240 |
| Offset and witness | 494 / 1,200 | 640 | 7,106 | 94 / 240 |

The development cohort selected a STOP offset of −0.25. Against the
unmodified policy it added 0.75 completion points on the primary cohort
(descriptive paired 95% interval 0.00 to +1.50), reduced premature STOP
by 18, and used 157 extra verifier calls. The one-edit witness changed
no in-family trajectories and used 2,248 extra verifier calls. It found
one actionable witness on the separate conflict cohort. None of the
observed gains reached the registered two-point completion target.

Retrospective replay found an unvisited feasible one-edit improvement
at only 3 of 658 in-family premature stops. There was a two-edit
improvement only at 488 stops, always a removal followed by an addition.
The other 167 had no feasible improvement within two unvisited edits.
These local rank improvements are possibilities, not certified complete
repairs. All 2,248 probes from the in-family witness arm were vetoed.

DataGoblin `e88bbe0` reproduced all three archived action-set checkpoint
digests and kept the original decision, proposal, and recurrent caps.
Independent audit replayed every logged policy check and all 6,840
development and frozen trials. The default inference path also reproduced
all actions and counters in 1,440 archived trials. The run took 49.3 CPU
wall seconds with 526.63 MiB peak memory.

## Bounded exchange witness

`python -m metta_research_bench.looped.stop_exchange --prior /path/to/action-ties-verified --stop-archive /path/to/stop-policy-run --out /path/to/stop-exchange-run`
tests whether replacing one record at a proposed STOP recovers completions
on fresh roots. The public prefilter screens budget, dependencies, lineage,
revision, eligibility, conflict and the endpoint rank. The policy proposes
only its best screened removal-addition pair. A typed verifier must accept
both edits, and MeTTa must certify that the receipts form a chain to a feasible
strictly better state. A failed exchange leaves the actor's selection intact.

The exchange costs two edit slots and reserves a final STOP slot. Its checks
share the four-proposal cap and the two-call witness budget. The intermediate
state is recorded and checked. The model makes no additional recurrent pass
between the two verified edits. The four arms cross the exchange guard with
the previously frozen STOP offset. The 400 new in-family roots and 80 new
conflict roots are separate from both earlier studies' training and scoring
cohorts, and the three checkpoints must reproduce their archived digests.

| STOP policy | In-family completions | Premature STOP | Verifier calls | Conflict completions |
| --- | ---: | ---: | ---: | ---: |
| Unmodified | 510 / 1,200 | 642 | 4,502 | 76 / 240 |
| Calibrated offset | 513 / 1,200 | 624 | 4,735 | 76 / 240 |
| Verified exchange | 816 / 1,200 | 327 | 5,511 | 159 / 240 |
| Offset and exchange | 800 / 1,200 | 323 | 5,729 | 159 / 240 |

The exchange gained 25.50 completion points in-family (descriptive 95%
paired interval +22.25 to +28.92) and 34.58 points in the conflict cohort
(+26.25 to +42.92). It recovered 306 and 83 failed trials respectively,
with no losses among trials the raw policy completed. Total in-family
verifier calls rose by 1,009 and recurrent visits rose by 1,665. Calls
per verified completion fell from 8.83 to 6.75. Both arms applied zero
unsafe additions under the shared gate.

The exchange-only arm screened 10,824 in-family pairs by public constraints
and spent 890 typed checks on 445 applied exchanges. It had no witness
vetoes on that cohort. The model weights and original caps were fixed.
The gain comes from the inference controller, and the total consumed
compute differs across arms. Independent audits replayed all 5,760
frozen trials and 1,115 native exchange certificates. DataGoblin
`3dcfdbd` ran in 52.6 CPU seconds with 506.26 MiB peak memory.
