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
the environment's typed proposal check and advance the repair by one
membership. At the verified terminal state the only target is STOP.
The native `action_ties.metta` rule reads environment receipts and
checks the exported target set. The exact Python verifier supplies
the endpoint and repair distances.

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
